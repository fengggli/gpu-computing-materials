/*
 * Description:
 *
 * Author: Yuankun Fu
 * e-mail: qoofyk@gmail.com
 */

#include "awnn/tensor.h"
#include "awnn/layer.h"
#include "awnn/layer_conv.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#define THRESHOLD               2.0e-2

#include <time.h>
static double second (void)
{
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return ((double)tp.tv_sec + (double)tp.tv_nsec / 1000000000.0);
}

//Generate uniform numbers [0,1)
static void initImage(T* image, int imageSize) {
  static unsigned seed = 123456789;
  for (int index = 0; index < imageSize; index++) {
    seed = ( 1103515245 * seed + 12345 ) & 0xffffffff;
    image[index] = float(seed)*2.3283064e-10; //2^-32
  }
}


static int checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
  if (code) {
    printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int) code, cudaGetErrorString(code), expr);
    return 1;
  }
  return 0;
}

static int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
  if (code)  {
    printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int) code, cudnnGetErrorString(code), expr);
    return 1;
  }
  return 0;
}

#define checkCudaErr(...)       do { int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); if (err) goto clean; } while (0)

#define checkCudnnErr(...)      do { int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  if (err) goto clean; } while (0)


static void printPerf( double cudaTime, double cudaGflops, double cudaBandwithGb,
                       const char *cpuLib, double cpuTime,  double cpuGflops, double cpuBandwithGb) {
  printf( "^^^^ CUDA : elapsed = %g sec,  ",  cudaTime );
  if (cudaGflops > 0)    printf( "Gflops = %.3f ",      cudaGflops );
  if (cudaBandwithGb > 0) printf( "Bandwidth = %.3f ",  cudaBandwithGb );
  printf( "\n");
  if (cpuLib) {
    printf( "^^^^%s : elapsed = %g sec, ",  cpuLib, cpuTime );
    if (cpuGflops > 0)    printf( "Gflops = %.3f ",      cpuGflops );
    if (cpuBandwithGb > 0) printf( "Bandwidth = %.3f, ",  cpuBandwithGb );
    printf( "Speedup %.2f\n",  cpuTime/cudaTime );

  }
}

static void generateStrides(const int* dimA, int* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
  //For INT8x4 and INT8x32 we still compute standard strides here to input
  //into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
  if (filterFormat == CUDNN_TENSOR_NCHW || filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
    strideA[nbDims-1] = 1 ;
    for(int d = nbDims-2 ; d >= 0 ; d--) {
      strideA[d] = strideA[d+1] * dimA[d+1] ;
    }
  } else {
    //Here we assume that the format is CUDNN_TENSOR_NHWC
    strideA[1] = 1;
    strideA[nbDims-1] = strideA[1]*dimA[1];
    for(int d = nbDims-2 ; d >= 2 ; d--) {
      strideA[d] = strideA[d+1] * dimA[d+1] ;
    }
    strideA[0] = strideA[2]*dimA[2];
  }
}

// Convert a linear index
// i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
// into a multidimensional index
// (d_1, d_2, ..., d_n)
void lin2dim(int id, int* ids, const int* dims, int length) {
  int idrem = id ;
  int prod  = 1 ; // accumulates the product of the dimensions
  for(int i = length-1; i >= 0; i--) {
    ids[i] = (idrem / prod) % dims[i] ;
    idrem = id - ids[i] * prod ;
    prod *= dims[i] ;
  }
}

// Convert a multidimensional index
// (d_1, d_2, ..., d_n)
// into a linear index
// i = d_1 s_1 + ... + d_n s_n
static int dim2lin(const int* ids, const int* strides, int length) {
  int res = 0 ;
  for(int i = 0 ; i < length ; i++) {
    res += ids[i] * strides[i];
  }
  return res ;
}

static T doFma(T fval, T ival, T tmp) {
  return fval*ival+tmp;
}

static void doEpilog(T *out, int idx, T alphaAcc, T beta) {
  if( beta == 0.f ) {
    out[idx] = alphaAcc;
  } else {
    out[idx] = alphaAcc + out[idx]*beta;
  }
}

//T_ELEM is the type the data is stored in, T_MATH is the type the calculations are done in.
template <typename T_ELEM, typename T_MATH>
static void conv_cpu_ref (
    const T_ELEM* inputData,
    const T_ELEM* filterData,
    T_ELEM*       outputData,
    float        alpha,
    float        beta,
    int          resizeFactor,
    cudnnTensorFormat_t filterFormat,
    const int*   inDims,
    const int*   filDims,
    const int*   outDims,
    const int*   inStride,
    const int*   outStride,
    const int*   stride,
    const int*   pad,
    const int*   dilation,
    int          nbDims
) {
  int imDims = nbDims - 2 ;

  int filStride[8] = {0} ;
  generateStrides(filDims, filStride, nbDims, filterFormat);

  bool isConv = true; //(CUDNN_CONVOLUTION == mode) ;
  // Number of pixels in output
  int nPixelsOut = 1 ;
  for(int i = 2 ; i < nbDims ; i++)
    nPixelsOut *= outDims[i] ;
  // Number of pixels in filter
  int nPixelsFil = 1 ;
  for(int i = 2 ; i < nbDims ; i++)
    nPixelsFil *= filDims[i] ;
  // Used to store coordinates
  int filIds[8] = {0} ;
  int outIds[8] = {0} ;
  int inIds [8] = {0} ;
  int tmpIds[8] = {0} ;

  // For each image in the output
  for(int ni = 0 ; ni < outDims[0] ; ni++) {
    // For each outer feature layer of the output image
    for(int ki_outer = 0 ; ki_outer < outDims[1] / resizeFactor; ki_outer++) {
      int outputOffset = ni * outStride[0] / resizeFactor + ki_outer * outStride[1] ;
      // For every pixel in this output image's feature layer
      for(int outId = 0 ; outId < nPixelsOut ; outId++) {
        // Get output pixel ids
        lin2dim(outId, outIds, outDims+2, imDims) ; // Skip n and k dimensions
        // Now we get the coordinates in input space of the "top left" corner of the filter: multiply by stride and remove pad
        for(int d = 0 ; d < imDims ; d++) {
          inIds[d] = outIds[d] * stride[d] - pad[d] ;
        }
        // For each inner feature layer of the output image
        for (int ki_inner = 0; ki_inner < resizeFactor; ki_inner++) {
          // We prepare to accumulate
          T_MATH tmp = 0;
          // For each outer feature layer of the input image and filter
          for(int ci = 0 ; ci < inDims[1] / resizeFactor; ci++) {
            int inputOffset = ni * inStride[0] / resizeFactor + ci * inStride[1] ;
            int filterOffset = (ki_outer * resizeFactor + ki_inner) * filStride[0] / resizeFactor + ci * filStride[1] ;
            // Now for every pixel in the filter
            for(int filId = 0 ; filId < nPixelsFil ; filId ++) {
              // Get the position of the pixel
              lin2dim(filId, filIds, filDims+2, imDims) ;
              // Compute the corresponding output pixel
              // and check whether we are in the padding area on the fly too (not that for convolution, we flip the image patch (equivalent to flipping the filter patch))
              bool inside = true ;
              for(int d = 0 ; d < imDims && inside ; d++) {
                if (isConv) {
                  tmpIds[d] = inIds[d] + dilation[d] * (filDims[2+d]-1 - filIds[d]) ;
                } else {
                  tmpIds[d] = inIds[d] + dilation[d] * filIds[d] ;
                }
                inside &= (tmpIds[d] >= 0 && tmpIds[d] < inDims[2+d]) ; // If we are in the padding area: stop and skip computations
              }
              if(inside) {
                int actualTmpId = inputOffset + dim2lin(tmpIds, (inStride)+2, imDims) ;
                //int actualFilId = filterOffset + filId ;
                int actualFilId = filterOffset + dim2lin(filIds, (filStride)+2, imDims) ;

                //For each inner feature layer of the input image and filter
                for (int i = 0; i < resizeFactor; i++) {
                  T_ELEM fval = filterData[actualFilId * resizeFactor + i] ;
                  T_ELEM ival = inputData[actualTmpId * resizeFactor + i] ;
                  tmp = doFma(fval, ival, tmp);
                }
              }
            }
          }

          // Store final result in proper position in output image
          int actualOutId = outputOffset + dim2lin(outIds, (outStride)+2, imDims) ;
          doEpilog(outputData, actualOutId * resizeFactor + ki_inner, alpha*tmp, beta);
        }



      }
    }
  }
}

template<typename T_ELEM>
static void dataGrad_cpu_ref (
    const T_ELEM *weight,
    const T_ELEM *top_diff,
    T_ELEM *output,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    const int*   inDims,
    const int*   filDims,
    const int*   outDims,
    const int*   inStride,
    const int*   outStride,
    const int*   stride,
    const int*   pad,
    const int*   dilation,
    int          nbDims )
{

  // Sanity checks
  // output is n x c x h x w
  // diff   is n x k x p x q
  // filter is k x c x r x s
  assert(inDims[0] == outDims[0]); // n
  assert(inDims[1] == filDims[0]); // k
  assert(outDims[1] == filDims[1]); // cactualOutId

  int filStride[8] = {0} ;
  generateStrides(filDims, filStride, nbDims, filterFormat);

  bool isConv = true; //(CUDNN_CONVOLUTION == mode) ;

  // For every output pixel (n x c x h x w)
  for(int ni = 0; ni < outDims[0]; ni++) {
    for(int ci = 0; ci < outDims[1]; ci++) {
      for(int hi = 0; hi < outDims[2]; hi++) {
        for(int wi = 0; wi < outDims[3]; wi++) {
          int outIdx = ni * outStride[0] +
                       ci * outStride[1] +
                       hi * outStride[2] +
                       wi * outStride[3];
          T val = 0.0;

          // For every diff channel (k)
          for(int ki = 0; ki < inDims[1]; ki++) { // Sum over k channels
            int offset_filter = ki * filStride[0] + ci * filStride[1];
            int offset_diff   = ni * inStride[0] + ki * inStride[1];
            // For every pixel if filter (r x s)
            for(int ri = 0; ri < filDims[2]; ri++) {
              int p = hi + pad[0];
              if (isConv){
                p -= (filDims[2] - 1 - ri) * dilation[0];
              } else {
                p -= ri * dilation[0];
              }
              if ( p%stride[0] )
                continue;
              p/=stride[0];

              for(int si = 0; si < filDims[3]; si++) {
                int q = wi + pad[1];
                // Fetch the value in filter and diff, product and accumulate
                // So basically, for the convolution, we replace r by dim-1-r and s by dim-1-s to "flip" the filter
                // We can then just reason in term of correlation
                if (isConv){
                  q -= (filDims[3] - 1 - si) * dilation[1];
                } else {
                  q -= si * dilation[1];
                }
                //Skip if q or p isn't multiple of strides
                if ( q%stride[1] )
                  continue;
                q/=stride[1];
                int inBounds = ( (p >= 0) && (p < inDims[2]) && (q >= 0) && (q < inDims[3]) );
                if (inBounds) {
                  int filterIdx = offset_filter + ri * filStride[2] + si * filStride[3];
                  int diffIdx = offset_diff + p * inStride[2] + q * inStride[3];
                  T_ELEM imTmp = top_diff[diffIdx];
                  T_ELEM filTmp = weight[filterIdx];
                  val = doFma(filTmp, imTmp, val);
                }
              }
            }
          }
          doEpilog(output, outIdx, alpha*val, beta);
        }
      }
    }
  }
}
//TODO:Fix this mess
template<typename T_ELEM>
static void weightGrad_cpu_ref(/*const TensorNdTestDesc_t *tensorInputDesc,*/
    const T_ELEM *image,
    /*const TensorNdTestDesc_t *tensorDiffDesc,*/
    const T_ELEM *diffData,
    /*const ConvNdTestDesc_t *convDesc,*/
    /*const TensorNdTestDesc_t *filterOutputDesc,*/
    float alpha,
    float beta,
    T_ELEM *output,
    cudnnTensorFormat_t filterFormat,
    const int*   inDims,
    const int*   filDims,
    const int*   diffDims,
    const int*   inStride,
    const int*   diffStride,
    const int*   stride,
    const int*   pad,
    const int*   dilation,
    int          nbDims )
{
  // Some sanity checks
  // image   is n x c x h x w
  // diff    is n x k x p x q
  // filter  is k x c x r x s
  assert(inDims[0] == diffDims[0]) ;
  assert(inDims[1] == filDims[1]) ;
  assert(diffDims[1]  == filDims[0]) ;

  // Filter stride
  int filterStride[4] ;
  generateStrides(filDims, filterStride, nbDims, filterFormat);

  bool isConv = true; //(CUDNN_CONVOLUTION == mode) ;

  // For every filter pixel (k x c x r x s)
  for(int ci = 0; ci < inDims[1]; ci++) { // Loop over filter output pixels
    for(int ri = 0; ri < filDims[2]; ri++) { //        ^
      for(int si = 0; si < filDims[3]; si++) { //    ^
        for(int ki = 0; ki < filDims[0]; ki++){ // ^
          int filIdx = ki * filterStride[0] + ci * filterStride[1] + ri * filterStride[2] + si * filterStride[3] ;
          T val = 0.f ;
          // For every image (n)
          for(int ni = 0 ; ni < inDims[0]; ni++) { // Sum over the batch
            int offset_image  = ni * inStride[0] + ci * inStride[1] ;
            int offset_diff   = ni * diffStride[0]  + ki * diffStride[1] ;
            // For every pixel in diff (p x q)
            for(int pi = 0; pi < diffDims[2] ; pi++ ) { // Sum over the pixels of diff
              for(int qi = 0; qi < diffDims[3] ; qi++ ) { //  ^
                // Fetch the value in image and diff, product and accumulate
                int y = pi * stride[0] - pad[0] ;
                int x = qi * stride[1] - pad[1] ;
                // Convolution = Correlation with a flipped filter
                // So basically, for the convolution, we replace r by dim-1-r and s by dim-1-s to "flip" the filter
                // We can then just reason in term of correlation
                if (isConv){
                  y += (filDims[2] - 1 - ri) * dilation[0] ;
                  x += (filDims[3] - 1 - si) * dilation[1] ;
                } else {
                  // The effect of dilation on the gradient is to start the "zone of influence" of a given pixel further into the image, so dilation
                  // only produces a shift in x and y
                  y += ri * dilation[0] ;
                  x += si * dilation[1] ;
                }
                // Image value
                int inBounds = ((x >=0)&&(x < inDims[3])&&(y >=0)&&(y < inDims[2]));
                if (inBounds) {
                  int imIdx    = offset_image  + y * inStride[2] + x * inStride[3] ;
                  // Diff value
                  int diffIdx  = offset_diff   + pi * diffStride[2]  + qi * diffStride[3] ;
                  // Prod and accumulate
                  T_ELEM imTmp = image[imIdx] ;
                  T_ELEM diffTmp = diffData[diffIdx];
                  val = doFma(diffTmp, imTmp, val);
                }
              }
            }
          }
          doEpilog(output, filIdx, alpha*val, beta);
        }
      }
    }
  }
}


T getError(T dev, T ref) {
  if (ref > 1.0 || ref < -1.0)
    return (dev - ref)/ref;
  else
    return dev - ref;
}

//float getError(half1 dev, half1 ref) {
//  if (cpu_half2float(ref) > 1.0 || cpu_half2float(ref) < -1.0)
//    return (cpu_half2float(dev) - cpu_half2float(ref))/cpu_half2float(ref);
//  else
//    return cpu_half2float(dev) - cpu_half2float(ref);
//}

//int8_t getError(int8_t dev, int8_t ref) {
//  return dev - ref;
//}

static inline int getFwdConvDilatedFilterDim(int filterDim,
                                             int dilation)
{
  return ( (filterDim - 1) * dilation ) + 1 ;
}

static inline int getFwdConvPaddedImageDim(int tensorDim,
                                           int pad)
{
  return tensorDim + (2 * pad) ;
}

static inline int getFwdConvOutputDim( int tensorDim,
                                       int pad,
                                       int filterDim,
                                       int stride,
                                       int dilation)
{
  int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation))/stride + 1;
  return(p);
}

template <typename T_ELEM>
int doConv(
    cudnnHandle_t handle_,
    T_ELEM* devPtrI,
    T_ELEM* devPtrF,
    T_ELEM* devPtrO,
    T_ELEM* hostI,
    T_ELEM* hostF,
    T_ELEM* hostO,
    cudnnTensorDescriptor_t cudnnIdesc,
    cudnnFilterDescriptor_t cudnnFdesc,
    cudnnTensorDescriptor_t cudnnOdesc,
    cudnnConvolutionDescriptor_t cudnnConvDesc,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    cudnnDataType_t dataType,
    const int*   dimA,
    const int*   filterdimA,
    const int*   outdimA,
    const int*   strideA,
    const int*   outstrideA,
    const int*   convstrideA,
    const int*   padA,
    const int*   dilationA,
    const int   benchmark) {

  int outsize = outstrideA[0]*outdimA[0];
  T_ELEM* hostOfromdev = (T_ELEM*)calloc (outsize, sizeof(hostO[0]) );

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  void *workSpace = 0;
  size_t workSpaceSize;
  int numErrors = 0;
  double start, stop;

  checkCudnnErr ( cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc,
                                                          cudnnOdesc, algo, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }
  start = second();
  checkCudnnErr ( cudnnConvolutionForward (handle_,
                                           (void*)(&alpha),
                                           cudnnIdesc, devPtrI,
                                           cudnnFdesc, devPtrF,
                                           cudnnConvDesc,
                                           algo,
                                           workSpace, workSpaceSize,
                                           (void*)(&beta),
                                           cudnnOdesc, devPtrO) );
  checkCudaErr( cudaDeviceSynchronize() );
  stop = second();
  printPerf( stop - start, 0, 0,
             0, 0, 0, 0);
  checkCudaErr( cudaMemcpy(hostOfromdev, devPtrO, sizeof(hostO[0]) * outsize, cudaMemcpyDeviceToHost) );
  checkCudaErr( cudaDeviceSynchronize() );

  if (!benchmark) {
    conv_cpu_ref<T_ELEM, T>( hostI, hostF, hostO, alpha, beta, 1, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);

    for (int index = 0; index < outsize; index++) {
      T diff = getError(hostOfromdev[index], hostO[index]);
      if (diff < 0) diff = -diff;
      if(diff > THRESHOLD) {
        numErrors++;
      }
      //printf("cuda result is %d, and reference is %d\n", hostOfromdev[index], hostO[index]);
    }
  }
  clean:
  if (hostOfromdev) free(hostOfromdev);
  if (workSpace) cudaFree(workSpace);
  return numErrors;
}

template <typename T_ELEM>
int doDgrad(
    cudnnHandle_t handle_,
    T_ELEM* devPtrI,
    T_ELEM* devPtrF,
    T_ELEM* devPtrO,
    T_ELEM* hostI,
    T_ELEM* hostF,
    T_ELEM* hostO,
    cudnnTensorDescriptor_t cudnnIdesc,
    cudnnFilterDescriptor_t   cudnnFdesc,
    cudnnTensorDescriptor_t cudnnOdesc,
    cudnnConvolutionDescriptor_t cudnnConvDesc,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    const int*   dimA,
    const int*   filterdimA,
    const int*   outdimA,
    const int*   strideA,
    const int*   outstrideA,
    const int*   convstrideA,
    const int*   padA,
    const int*   dilationA,
    const int    benchmark) {

  int insize = strideA[0]*dimA[0];
  T_ELEM* hostIfromdev = (T_ELEM*)calloc (insize, sizeof(hostI[0]) );
  cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  void *workSpace = 0;
  size_t workSpaceSize;
  int numErrors = 0;
  double start, stop;

  checkCudnnErr ( cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc,
                                                               cudnnIdesc, algo, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }
  start = second();
  checkCudnnErr ( cudnnConvolutionBackwardData (handle_,
                                                (void*)(&alpha),
                                                cudnnFdesc, devPtrF,
                                                cudnnOdesc, devPtrO,
                                                cudnnConvDesc,
                                                algo,
                                                workSpace, workSpaceSize,
                                                (void*)(&beta),
                                                cudnnIdesc, devPtrI) );
  checkCudaErr( cudaDeviceSynchronize() );
  stop = second();
  printPerf( stop - start, 0, 0,
             0, 0, 0, 0);
  checkCudaErr( cudaMemcpy(hostIfromdev, devPtrI, sizeof(hostI[0]) * insize, cudaMemcpyDeviceToHost) );
  checkCudaErr( cudaDeviceSynchronize() );

  if (!benchmark) {
    dataGrad_cpu_ref<T_ELEM>(hostF, hostO, hostI,  alpha, beta, filterFormat, outdimA, filterdimA, dimA, outstrideA, strideA, convstrideA, padA, dilationA, 4);
    for (int index = 0; index < insize; index++) { // assuming in data is packed
      T diff = getError(hostIfromdev[index], hostI[index]);
      if (diff < 0) diff = -diff;
      if(diff > THRESHOLD) {
        numErrors++;
      }
    }
  }
  clean:
  if (hostIfromdev) free(hostIfromdev);
  if (workSpace) cudaFree(workSpace);
  return numErrors;
}

template <typename T_ELEM>
int doWgrad(
    cudnnHandle_t handle_,
    T_ELEM* devPtrI,
    T_ELEM* devPtrF,
    T_ELEM* devPtrO,
    T_ELEM* hostI,
    T_ELEM* hostF,
    T_ELEM* hostO,
    cudnnTensorDescriptor_t cudnnIdesc,
    cudnnFilterDescriptor_t cudnnFdesc,
    cudnnTensorDescriptor_t cudnnOdesc,
    cudnnConvolutionDescriptor_t cudnnConvDesc,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    const int*   dimA,
    const int*   filterdimA,
    const int*   outdimA,
    const int*   strideA,
    const int*   outstrideA,
    const int*   convstrideA,
    const int*   padA,
    const int*   dilationA,
    const int   benchmark) {

  int filsize = filterdimA[0]*filterdimA[1]*filterdimA[2]*filterdimA[3];
  T_ELEM* hostFfromdev = (T_ELEM*)calloc (filsize, sizeof(hostF[0]) );
  cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  void *workSpace = 0;
  size_t workSpaceSize;
  int numErrors = 0;
  double start, stop;

  checkCudnnErr ( cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_, cudnnIdesc, cudnnOdesc, cudnnConvDesc,
                                                                 cudnnFdesc, algo, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }
  start = second();
  checkCudnnErr ( cudnnConvolutionBackwardFilter (handle_,
                                                  (void*)(&alpha),
                                                  cudnnIdesc, devPtrI,
                                                  cudnnOdesc, devPtrO,
                                                  cudnnConvDesc,
                                                  algo,
                                                  workSpace, workSpaceSize,
                                                  (void*)(&beta),
                                                  cudnnFdesc, devPtrF) );
  checkCudaErr( cudaDeviceSynchronize() );
  stop = second();
  printPerf( stop - start, 0, 0,
             0, 0, 0, 0);
  checkCudaErr( cudaMemcpy(hostFfromdev, devPtrF, sizeof(hostF[0]) * filsize, cudaMemcpyDeviceToHost) );
  checkCudaErr( cudaDeviceSynchronize() );

  if (!benchmark) {
    weightGrad_cpu_ref<T_ELEM>(hostI, hostO, alpha, beta, hostF, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
    for (int index = 0; index < filsize; index++) { // assuming in data is packed
      T diff = getError(hostFfromdev[index], hostF[index]);
      if (diff < 0) diff = -diff;
      if(diff > THRESHOLD) {
        numErrors++;
      }
    }
  }
  clean:
  if (hostFfromdev) free(hostFfromdev);
  if (workSpace) cudaFree(workSpace);
  return numErrors;
}

template <typename T_ELEM>
status_t doTest(int algo, int* dimA, int* padA, int* convstrideA, int* filterdimA, cudnnTensorFormat_t filterFormat, cudnnDataType_t dataType, int mathType, int benchmark) {

  cudnnHandle_t handle_;
  T_ELEM* devPtrI=NULL;
  T_ELEM* devPtrF=NULL;
  T_ELEM* devPtrO=NULL;
  T_ELEM* hostI=NULL;
  T_ELEM* hostF=NULL;
  T_ELEM* hostO=NULL;

  cudnnTensorDescriptor_t cudnnIdesc;
  cudnnFilterDescriptor_t cudnnFdesc;
  cudnnTensorDescriptor_t cudnnOdesc;
  cudnnConvolutionDescriptor_t cudnnConvDesc;

  int convDim = 2;

  float alpha = 1.0f;
  float beta = 0.0;
  int numErrors = 0;
  int dilationA[] = {1, 1};
  int insize = 0;
  int filtersize = 0;
  int outdimA[] = {1, 8, 30, 30};
  int outsize = 0;

  int dimA_padded[4];
  int outdimA_padded[4];
  int filterdimA_padded[4];
  int strideA_padded[4];
  int outstrideA_padded[4];
  int filterstrideA_padded[4];

  outdimA[0] = dimA[0];
  outdimA[1] = filterdimA[0];
  for( int dim = 0; dim < 2; dim++) {
    outdimA[dim+2] = getFwdConvOutputDim( dimA[dim+2],
                                          padA[dim],
                                          filterdimA[dim+2],
                                          convstrideA[dim],
                                          dilationA[dim]);
  }

  for (int i = 0; i < 4; i++) {
    dimA_padded[i] = dimA[i];
    outdimA_padded[i] = outdimA[i];
    filterdimA_padded[i] = filterdimA[i];
  }

  printf("====USER DIMENSIONS====\n");
  printf("input dims are %d, %d, %d, %d\n", dimA[0], dimA[1], dimA[2], dimA[3]);
  printf("filter dims are %d, %d, %d, %d\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
  printf("output dims are %d, %d, %d, %d\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);
  printf("====PADDING DIMENSIONS====\n");
  printf("padded input dims are %d, %d, %d, %d\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
  printf("padded filter dims are %d, %d, %d, %d\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
  printf("padded output dims are %d, %d, %d, %d\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);


  checkCudnnErr(cudnnCreate(&handle_));

  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
  checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
  checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

  generateStrides(dimA_padded, strideA_padded, 4, filterFormat);
  insize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];

  generateStrides(filterdimA_padded, filterstrideA_padded, 4, filterFormat);
  filtersize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];

  generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);
  outsize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

  cudaMalloc ((void**)&(devPtrI), (insize) * sizeof(devPtrI[0]) );
  cudaMalloc ((void**)&(devPtrF), (filtersize) * sizeof(devPtrF[0]) );
  cudaMalloc ((void**)&(devPtrO), (outsize) * sizeof(devPtrO[0]) );
  hostI = (T_ELEM*)calloc (insize, sizeof(hostI[0]) );
  hostF = (T_ELEM*)calloc (filtersize, sizeof(hostF[0]) );
  hostO = (T_ELEM*)calloc (outsize, sizeof(hostO[0]) );

  initImage(hostI, insize);
  initImage(hostF, filtersize);
  initImage(hostO, outsize);


  checkCudaErr( cudaMemcpy(devPtrI, hostI, sizeof(hostI[0]) * insize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaMemcpy(devPtrF, hostF, sizeof(hostF[0]) * filtersize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaMemcpy(devPtrO, hostO, sizeof(hostO[0]) * outsize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaDeviceSynchronize() );


  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim+2, dimA_padded, strideA_padded) );
  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim+2, outdimA_padded, outstrideA_padded) );
  checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                                 convDim,
                                                 padA,
                                                 convstrideA,
                                                 dilationA,
                                                 CUDNN_CONVOLUTION,
                                                 CUDNN_DATA_FLOAT) );

  checkCudnnErr( cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, filterFormat, convDim+2, filterdimA_padded));

  if (mathType == 1) {
    checkCudnnErr( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );
  }

  if (algo == 0) {
    printf("Testing conv\n");
    numErrors = doConv(
        handle_,
        devPtrI,
        devPtrF,
        devPtrO,
        hostI,
        hostF,
        hostO,
        cudnnIdesc,
        cudnnFdesc,
        cudnnOdesc,
        cudnnConvDesc,
        alpha,
        beta,
        filterFormat,
        dataType,
        dimA_padded,
        filterdimA_padded,
        outdimA_padded,
        strideA_padded,
        outstrideA_padded,
        convstrideA,
        padA,
        dilationA,
        benchmark);
  } else if (algo == 1) {
    printf("Testing dgrad\n");
    numErrors = doDgrad(
        handle_,
        devPtrI,
        devPtrF,
        devPtrO,
        hostI,
        hostF,
        hostO,
        cudnnIdesc,
        cudnnFdesc,
        cudnnOdesc,
        cudnnConvDesc,
        alpha,
        beta,
        filterFormat,
        dimA,
        filterdimA,
        outdimA,
        strideA_padded,
        outstrideA_padded,
        convstrideA,
        padA,
        dilationA,
        benchmark);
  } else {
    printf("Testing wgrad\n");
    numErrors = doWgrad(
        handle_,
        devPtrI,
        devPtrF,
        devPtrO,
        hostI,
        hostF,
        hostO,
        cudnnIdesc,
        cudnnFdesc,
        cudnnOdesc,
        cudnnConvDesc,
        alpha,
        beta,
        filterFormat,
        dimA,
        filterdimA,
        outdimA,
        strideA_padded,
        outstrideA_padded,
        convstrideA,
        padA,
        dilationA,
        benchmark);
  }

  if (!benchmark) {
    if (numErrors == 0) {
      printf("Test PASSED\n");
    } else {
      printf("Test FAILED, num errors = %d\n", numErrors);
    }
  }

  clean:
  if (devPtrI) cudaFree (devPtrI);
  if (devPtrF) cudaFree (devPtrF);
  if (devPtrO) cudaFree (devPtrO);
  if (hostI) free(hostI);
  if (hostF) free(hostF);
  if (hostO) free(hostO);
  if (cudnnIdesc) cudnnDestroyTensorDescriptor(cudnnIdesc);
  if (cudnnFdesc) cudnnDestroyFilterDescriptor(cudnnFdesc);
  if (cudnnOdesc) cudnnDestroyTensorDescriptor(cudnnOdesc);
  if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
  if (handle_) cudnnDestroy(handle_);

  return S_OK;
}

template <typename T_ELEM>
status_t doForward(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y,
    int* dimA, int* padA, int* convstrideA, int* filterdimA, cudnnTensorFormat_t filterFormat, cudnnDataType_t dataType, int mathType, int benchmark) {

  cudnnHandle_t handle_;
  T_ELEM* devPtrI=NULL;
  T_ELEM* devPtrF=NULL;
  T_ELEM* devPtrO=NULL;
  T_ELEM* hostI=NULL;
  T_ELEM* hostF=NULL;
  T_ELEM* hostO=NULL;

  cudnnTensorDescriptor_t cudnnIdesc;
  cudnnFilterDescriptor_t cudnnFdesc;
  cudnnTensorDescriptor_t cudnnOdesc;
  cudnnConvolutionDescriptor_t cudnnConvDesc;

  int convDim = 2;

  float alpha = 1.0f;
  float beta = 0.0;
  int numErrors = 0;
  int dilationA[] = {1, 1};
  int insize = 0;
  int filtersize = 0;
  int outdimA[] = {1, 8, 30, 30}; //put a random value here.
  int outsize = 0;

  int dimA_padded[4];
  int outdimA_padded[4];
  int filterdimA_padded[4];
  int strideA_padded[4];
  int outstrideA_padded[4];
  int filterstrideA_padded[4];

  outdimA[0] = dimA[0];
  outdimA[1] = filterdimA[0];
  for( int dim = 0; dim < 2; dim++) {
    outdimA[dim+2] = getFwdConvOutputDim( dimA[dim+2],
                                          padA[dim],
                                          filterdimA[dim+2],
                                          convstrideA[dim],
                                          dilationA[dim]);
  }

  for (int i = 0; i < 4; i++) {
    dimA_padded[i] = dimA[i];
    outdimA_padded[i] = outdimA[i];
    filterdimA_padded[i] = filterdimA[i];
  }

  printf("====USER DIMENSIONS====\n");
  printf("input dims are %d, %d, %d, %d\n", dimA[0], dimA[1], dimA[2], dimA[3]);
  printf("filter dims are %d, %d, %d, %d\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
  printf("output dims are %d, %d, %d, %d\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);
  printf("====PADDING DIMENSIONS====\n");
  printf("padded input dims are %d, %d, %d, %d\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
  printf("padded filter dims are %d, %d, %d, %d\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
  printf("padded output dims are %d, %d, %d, %d\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);


  checkCudnnErr(cudnnCreate(&handle_));

  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
  checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
  checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

  generateStrides(dimA_padded, strideA_padded, 4, filterFormat);
  insize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];

  generateStrides(filterdimA_padded, filterstrideA_padded, 4, filterFormat);
  filtersize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];

  generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);
  outsize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

  cudaMalloc ((void**)&(devPtrI), (insize) * sizeof(devPtrI[0]) );
  cudaMalloc ((void**)&(devPtrF), (filtersize) * sizeof(devPtrF[0]) );
  cudaMalloc ((void**)&(devPtrO), (outsize) * sizeof(devPtrO[0]) );

  hostI = (T_ELEM*) (x.data);
  hostF = (T_ELEM*) (w.data);
  hostO = (T_ELEM*) (y.data);

#if 0
  hostI = (T_ELEM*)calloc (insize, sizeof(hostI[0]) );
  hostF = (T_ELEM*)calloc (filtersize, sizeof(hostF[0]) );
  hostO = (T_ELEM*)calloc (outsize, sizeof(hostO[0]) );

  initImage(hostI, insize);
  initImage(hostF, filtersize);
  initImage(hostO, outsize);
#endif

  checkCudaErr( cudaMemcpy(devPtrI, hostI, sizeof(hostI[0]) * insize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaMemcpy(devPtrF, hostF, sizeof(hostF[0]) * filtersize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaMemcpy(devPtrO, hostO, sizeof(hostO[0]) * outsize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaDeviceSynchronize() );


  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim+2, dimA_padded, strideA_padded) );
  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim+2, outdimA_padded, outstrideA_padded) );
  checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                                 convDim,
                                                 padA,
                                                 convstrideA,
                                                 dilationA,
                                                 CUDNN_CONVOLUTION,
                                                 CUDNN_DATA_FLOAT) );

  checkCudnnErr( cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, filterFormat, convDim+2, filterdimA_padded));

  if (mathType == 1) {
    checkCudnnErr( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );
  }


  printf("Testing conv\n");
  numErrors = doConv(
      handle_,
      devPtrI,
      devPtrF,
      devPtrO,
      hostI,
      hostF,
      hostO,
      cudnnIdesc,
      cudnnFdesc,
      cudnnOdesc,
      cudnnConvDesc,
      alpha,
      beta,
      filterFormat,
      dataType,
      dimA_padded,
      filterdimA_padded,
      outdimA_padded,
      strideA_padded,
      outstrideA_padded,
      convstrideA,
      padA,
      dilationA,
      benchmark);

  if (!benchmark) {
    if (numErrors == 0) {
      printf("Test PASSED\n");
    } else {
      printf("Test FAILED, num errors = %d\n", numErrors);
    }
  }

  clean:
  if (devPtrI) cudaFree (devPtrI);
  if (devPtrF) cudaFree (devPtrF);
  if (devPtrO) cudaFree (devPtrO);
  if (hostI) free(hostI);
  if (hostF) free(hostF);
  if (hostO) free(hostO);
  if (cudnnIdesc) cudnnDestroyTensorDescriptor(cudnnIdesc);
  if (cudnnFdesc) cudnnDestroyFilterDescriptor(cudnnFdesc);
  if (cudnnOdesc) cudnnDestroyTensorDescriptor(cudnnOdesc);
  if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
  if (handle_) cudnnDestroy(handle_);

  return S_OK;
}

template <typename T_ELEM>
status_t doBackward(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout,
    int* dimA, int* padA, int* convstrideA, int* filterdimA, cudnnTensorFormat_t filterFormat, cudnnDataType_t dataType, int mathType, int benchmark) {

  cudnnHandle_t handle_;
  T_ELEM* devPtrI=NULL;
  T_ELEM* devPtrF=NULL;
  T_ELEM* devPtrO=NULL;
  T_ELEM* hostI=NULL;
  T_ELEM* hostF=NULL;
  T_ELEM* hostO=NULL;

  cudnnTensorDescriptor_t cudnnIdesc;
  cudnnFilterDescriptor_t cudnnFdesc;
  cudnnTensorDescriptor_t cudnnOdesc;
  cudnnConvolutionDescriptor_t cudnnConvDesc;

  int convDim = 2;

  float alpha = 1.0f;
  float beta = 0.0;
  int numErrors = 0;
  int dilationA[] = {1, 1};
  int insize = 0;
  int filtersize = 0;
  int outdimA[] = {1, 8, 30, 30};
  int outsize = 0;

  int dimA_padded[4];
  int outdimA_padded[4];
  int filterdimA_padded[4];
  int strideA_padded[4];
  int outstrideA_padded[4];
  int filterstrideA_padded[4];

  outdimA[0] = dimA[0];
  outdimA[1] = filterdimA[0];
  for( int dim = 0; dim < 2; dim++) {
    outdimA[dim+2] = getFwdConvOutputDim( dimA[dim+2],
                                          padA[dim],
                                          filterdimA[dim+2],
                                          convstrideA[dim],
                                          dilationA[dim]);
  }

  for (int i = 0; i < 4; i++) {
    dimA_padded[i] = dimA[i];
    outdimA_padded[i] = outdimA[i];
    filterdimA_padded[i] = filterdimA[i];
  }

  printf("====USER DIMENSIONS====\n");
  printf("input dims are %d, %d, %d, %d\n", dimA[0], dimA[1], dimA[2], dimA[3]);
  printf("filter dims are %d, %d, %d, %d\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
  printf("output dims are %d, %d, %d, %d\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);
  printf("====PADDING DIMENSIONS====\n");
  printf("padded input dims are %d, %d, %d, %d\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
  printf("padded filter dims are %d, %d, %d, %d\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
  printf("padded output dims are %d, %d, %d, %d\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);


  checkCudnnErr(cudnnCreate(&handle_));

  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
  checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
  checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

  generateStrides(dimA_padded, strideA_padded, 4, filterFormat);
  insize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];

  generateStrides(filterdimA_padded, filterstrideA_padded, 4, filterFormat);
  filtersize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];

  generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);
  outsize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

  cudaMalloc ((void**)&(devPtrI), (insize) * sizeof(devPtrI[0]) );
  cudaMalloc ((void**)&(devPtrF), (filtersize) * sizeof(devPtrF[0]) );
  cudaMalloc ((void**)&(devPtrO), (outsize) * sizeof(devPtrO[0]) );


  hostI = (T_ELEM*) (dx.data);
  hostF = (T_ELEM*) (dw.data);
  hostO = (T_ELEM*) (dout.data);

#if 0
  hostI = (T_ELEM*)calloc (insize, sizeof(hostI[0]) );
  hostF = (T_ELEM*)calloc (filtersize, sizeof(hostF[0]) );
  hostO = (T_ELEM*)calloc (outsize, sizeof(hostO[0]) );

  initImage(hostI, insize);
  initImage(hostF, filtersize);
  initImage(hostO, outsize);
#endif

  checkCudaErr( cudaMemcpy(devPtrI, hostI, sizeof(hostI[0]) * insize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaMemcpy(devPtrF, hostF, sizeof(hostF[0]) * filtersize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaMemcpy(devPtrO, hostO, sizeof(hostO[0]) * outsize, cudaMemcpyHostToDevice));
  checkCudaErr( cudaDeviceSynchronize() );


  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim+2, dimA_padded, strideA_padded) );
  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim+2, outdimA_padded, outstrideA_padded) );
  checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                                 convDim,
                                                 padA,
                                                 convstrideA,
                                                 dilationA,
                                                 CUDNN_CONVOLUTION,
                                                 CUDNN_DATA_FLOAT) );

  checkCudnnErr( cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, filterFormat, convDim+2, filterdimA_padded));

  if (mathType == 1) {
    checkCudnnErr( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );
  }

  printf("Testing dgrad\n");
  numErrors = doDgrad(
      handle_,
      devPtrI,
      devPtrF,
      devPtrO,
      hostI,
      hostF,
      hostO,
      cudnnIdesc,
      cudnnFdesc,
      cudnnOdesc,
      cudnnConvDesc,
      alpha,
      beta,
      filterFormat,
      dimA,
      filterdimA,
      outdimA,
      strideA_padded,
      outstrideA_padded,
      convstrideA,
      padA,
      dilationA,
      benchmark);

  printf("Testing wgrad\n");
  numErrors = doWgrad(
      handle_,
      devPtrI,
      devPtrF,
      devPtrO,
      hostI,
      hostF,
      hostO,
      cudnnIdesc,
      cudnnFdesc,
      cudnnOdesc,
      cudnnConvDesc,
      alpha,
      beta,
      filterFormat,
      dimA,
      filterdimA,
      outdimA,
      strideA_padded,
      outstrideA_padded,
      convstrideA,
      padA,
      dilationA,
      benchmark);

  if (!benchmark) {
    if (numErrors == 0) {
      printf("Test PASSED\n");
    } else {
      printf("Test FAILED, num errors = %d\n", numErrors);
    }
  }

  clean:
  if (devPtrI) cudaFree (devPtrI);
  if (devPtrF) cudaFree (devPtrF);
  if (devPtrO) cudaFree (devPtrO);
  if (hostI) free(hostI);
  if (hostF) free(hostF);
  if (hostO) free(hostO);
  if (cudnnIdesc) cudnnDestroyTensorDescriptor(cudnnIdesc);
  if (cudnnFdesc) cudnnDestroyFilterDescriptor(cudnnFdesc);
  if (cudnnOdesc) cudnnDestroyTensorDescriptor(cudnnOdesc);
  if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
  if (handle_) cudnnDestroy(handle_);

  return S_OK;
}

status_t convolution_forward_cudnn(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y){

  int mathType = 0;
  int benchmark = 0;
//  int dimA[] = {1, 32, 4, 4};  // N, C, H, W;
  int dimA[] = {x.dim.dims[0], x.dim.dims[1], x.dim.dims[2], x.dim.dims[3]};  // N, C, H, W;
  int padA[] = {(int)params.padding, (int)params.padding};
  int convstrideA[] = {(int)params.stride, (int)params.stride};
  // batch size and feature layers must be multiples of 4 or 32 when using int8x4 or int8x32 respectively
  int filterdimA[] = {w.dim.dims[0], w.dim.dims[1], w.dim.dims[2], w.dim.dims[3]}; //k, c, r, s
//  int filterdimA[] = {32, 32, 1, 1}; //k, c, r, s
  cudnnTensorFormat_t  filterFormat = CUDNN_TENSOR_NCHW;

#if 0
  int device;
  struct cudaDeviceProp devProp;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&devProp, device);
  int deviceVer = devProp.major * 10 + devProp.minor;
#endif

  printf("Testing using cudnn forward\n");
//  int algo = 0;
//  status_t ret = doTest<T>(algo, dimA, padA, convstrideA, filterdimA, filterFormat, CUDNN_DATA_FLOAT, mathType, benchmark);

  status_t ret = doForward<T>(x, w, cache, params, y,
      dimA, padA, convstrideA, filterdimA, filterFormat, CUDNN_DATA_FLOAT, mathType, benchmark);

  return ret;
}

status_t convolution_backward_cudnn(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout){
  int mathType = 0;
  int benchmark = 0;
//  int dimA[] = {1, 32, 4, 4};  // N, C, H, W;
  int dimA[] = {dx.dim.dims[0], dx.dim.dims[1], dx.dim.dims[2], dx.dim.dims[3]};  // N, C, H, W;
  int padA[] = {(int)params.padding, (int)params.padding};
  int convstrideA[] = {(int)params.stride, (int)params.stride};
  //batch size and feature layers must be multiples of 4 or 32 when using int8x4 or int8x32 respectively
  int filterdimA[] = {dw.dim.dims[0], dw.dim.dims[1], dw.dim.dims[2], dw.dim.dims[3]}; //k, c, r, s
//  int filterdimA[] = {32, 32, 1, 1}; //k, c, r, s
  cudnnTensorFormat_t  filterFormat = CUDNN_TENSOR_NCHW;

  printf("Testing using cudnn backward\n");

  status_t ret = doBackward<T>(dx, dw, cache, params, dout,
      dimA, padA, convstrideA, filterdimA, filterFormat, CUDNN_DATA_FLOAT, mathType, benchmark);

  return ret;
}
