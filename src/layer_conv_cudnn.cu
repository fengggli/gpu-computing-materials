/*
 * Description:
 *
 * Author: Yuankun Fu
 * e-mail: qoofyk@gmail.com
 */

#include "awnn/tensor.h"
#include "awnn/layer.h"
#include "awnn/layer_conv.h"

#include "awnn/layer_cudnn.h"
#define THRESHOLD               2.0e-2

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
status_t doForward(tensor_t const x, tensor_t const w, tensor_t y, int* dimA,
                   int* padA, int* convstrideA, int* filterdimA, cudnnTensorFormat_t filterFormat, cudnnDataType_t dataType,
                   int mathType,
                   cudnnHandle_t handle_, cudnnTensorDescriptor_t cudnnIdesc,
                   cudnnFilterDescriptor_t cudnnFdesc,
                   cudnnTensorDescriptor_t cudnnOdesc,
                   cudnnConvolutionDescriptor_t cudnnConvDesc) {
//  cudnnHandle_t handle_;
  T_ELEM* devPtrI=x.data;
  T_ELEM* devPtrF=w.data;
  T_ELEM* devPtrO=y.data;

//  cudnnTensorDescriptor_t cudnnIdesc;
//  cudnnFilterDescriptor_t cudnnFdesc;
//  cudnnTensorDescriptor_t cudnnOdesc;
//  cudnnConvolutionDescriptor_t cudnnConvDesc;

  void *workSpace = 0;
  size_t workSpaceSize;

  /*cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;*/
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  /*cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;*/
  /*cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;*/

  int convDim = 2;

  float alpha = 1.0f;
  float beta = 0.0;
  int dilationA[] = {1, 1};
  int outdimA[4];


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

#ifdef PRINT_VERBOSE
  PDBG("====USER DIMENSIONS====\n");
  PDBG("input dims are %d, %d, %d, %d\n", dimA[0], dimA[1], dimA[2], dimA[3]);
  PDBG("filter dims are %d, %d, %d, %d\n", filterdimA[0], filterdimA[1],
       filterdimA[2], filterdimA[3]);
  PDBG("output dims are %d, %d, %d, %d\n", outdimA[0], outdimA[1], outdimA[2],
       outdimA[3]);
  PDBG("====PADDING DIMENSIONS====\n");
  PDBG("padded input dims are %d, %d, %d, %d\n", dimA_padded[0], dimA_padded[1],
       dimA_padded[2], dimA_padded[3]);
  PDBG("padded filter dims are %d, %d, %d, %d\n", filterdimA_padded[0],
       filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
  PDBG("padded output dims are %d, %d, %d, %d\n", outdimA_padded[0],
       outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);
#endif

//  checkCudnnErr(cudnnCreate(&handle_));
//
//  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
//  checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
//  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
//  checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

  generateStrides(dimA_padded, strideA_padded, 4, filterFormat);

  generateStrides(filterdimA_padded, filterstrideA_padded, 4, filterFormat);

  generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);

  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim+2, dimA_padded, strideA_padded) );
  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim+2, outdimA_padded, outstrideA_padded) );
  checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                                 convDim,
                                                 padA,
                                                 convstrideA,
                                                 dilationA,
                                                 CUDNN_CONVOLUTION, dataType));

  checkCudnnErr( cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, filterFormat, convDim+2, filterdimA_padded));

  if (mathType == 1) {
    checkCudnnErr( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );
  }

  // start computation of cudnn forward
  checkCudnnErr ( cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc,
                                                          cudnnOdesc, algo, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }

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

//clean:
//  if (cudnnIdesc) cudnnDestroyTensorDescriptor(cudnnIdesc);
//  if (cudnnFdesc) cudnnDestroyFilterDescriptor(cudnnFdesc);
//  if (cudnnOdesc) cudnnDestroyTensorDescriptor(cudnnOdesc);
//  if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
//  if (handle_) cudnnDestroy(handle_);

clean:
  if (workSpace) cudaFree(workSpace);

  return S_OK;
}

template <typename T_ELEM>
status_t doBackward(tensor_t x, tensor_t dx, tensor_t w, tensor_t dw,
                    tensor_t const dout, int* dimA, int* padA, int* convstrideA, int* filterdimA,
                    cudnnTensorFormat_t filterFormat,
                    cudnnDataType_t dataType, int mathType
                    ) {
  cudnnHandle_t handle_;
  T_ELEM* devPtr_dx = dx.data;
  T_ELEM* devPtr_w = w.data;

  T_ELEM* devPtr_x = x.data;
  T_ELEM* devPtr_dw = dw.data;

  T_ELEM* devPtrO = dout.data;

  cudnnTensorDescriptor_t cudnnIdesc;
  cudnnFilterDescriptor_t cudnnFdesc;
  cudnnTensorDescriptor_t cudnnOdesc;
  cudnnConvolutionDescriptor_t cudnnConvDesc;

  cudnnConvolutionBwdDataAlgo_t algo_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  cudnnConvolutionBwdFilterAlgo_t algo_weight = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  void *workSpace = 0;
  size_t workSpaceSize;

  int convDim = 2;

  float alpha = 1.0f;
  float beta = 0.0;
  int dilationA[] = {1, 1};
  int outdimA[4];

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

#ifdef PRINT_VERBOSE
  PDBG("====USER DIMENSIONS====\n");
  PDBG("input dims are %d, %d, %d, %d\n", dimA[0], dimA[1], dimA[2], dimA[3]);
  PDBG("filter dims are %d, %d, %d, %d\n", filterdimA[0], filterdimA[1],
       filterdimA[2], filterdimA[3]);
  PDBG("output dims are %d, %d, %d, %d\n", outdimA[0], outdimA[1], outdimA[2],
       outdimA[3]);
  PDBG("====PADDING DIMENSIONS====\n");
  PDBG("padded input dims are %d, %d, %d, %d\n", dimA_padded[0], dimA_padded[1],
       dimA_padded[2], dimA_padded[3]);
  PDBG("padded filter dims are %d, %d, %d, %d\n", filterdimA_padded[0],
       filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
  PDBG("padded output dims are %d, %d, %d, %d\n", outdimA_padded[0],
       outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);
#endif

  checkCudnnErr(cudnnCreate(&handle_));

  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
  checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
  checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
  checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

  generateStrides(dimA_padded, strideA_padded, 4, filterFormat);

  generateStrides(filterdimA_padded, filterstrideA_padded, 4, filterFormat);

  generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);


  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim+2, dimA_padded, strideA_padded) );
  checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim+2, outdimA_padded, outstrideA_padded) );
  checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                                 convDim,
                                                 padA,
                                                 convstrideA,
                                                 dilationA,
                                                 CUDNN_CONVOLUTION, dataType));

  checkCudnnErr( cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, filterFormat, convDim+2, filterdimA_padded));

  if (mathType == 1) {
    checkCudnnErr( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );
  }

  // start compute cudnn backward data
  checkCudnnErr ( cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc,
                                                               cudnnIdesc, algo_data, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }

  checkCudnnErr ( cudnnConvolutionBackwardData (handle_,
                                                (void*)(&alpha),
                                                cudnnFdesc, devPtr_w,
                                                cudnnOdesc, devPtrO,
                                                cudnnConvDesc,
                                                algo_data,
                                                workSpace, workSpaceSize,
                                                (void*)(&beta),
                                                cudnnIdesc, devPtr_dx) );
  checkCudaErr( cudaDeviceSynchronize() );

  // start compute cudnn backward filter
  checkCudnnErr ( cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_, cudnnIdesc, cudnnOdesc, cudnnConvDesc,
                                                                 cudnnFdesc, algo_weight, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }

  checkCudnnErr ( cudnnConvolutionBackwardFilter (handle_,
                                                  (void*)(&alpha),
                                                  cudnnIdesc, devPtr_x,
                                                  cudnnOdesc, devPtrO,
                                                  cudnnConvDesc,
                                                  algo_weight,
                                                  workSpace, workSpaceSize,
                                                  (void*)(&beta),
                                                  cudnnFdesc, devPtr_dw) );
  checkCudaErr( cudaDeviceSynchronize() );


clean:
  if (cudnnIdesc) cudnnDestroyTensorDescriptor(cudnnIdesc);
  if (cudnnFdesc) cudnnDestroyFilterDescriptor(cudnnFdesc);
  if (cudnnOdesc) cudnnDestroyTensorDescriptor(cudnnOdesc);
  if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
  if (handle_) cudnnDestroy(handle_);
  if (workSpace) cudaFree(workSpace);
  return S_OK;
}


status_t convolution_forward_cudnn(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y,
                                   cudnnHandle_t handle_, cudnnTensorDescriptor_t cudnnIdesc,
                                   cudnnFilterDescriptor_t cudnnFdesc,
                                   cudnnTensorDescriptor_t cudnnOdesc,
                                   cudnnConvolutionDescriptor_t cudnnConvDesc){
  int mathType = 0;  // 0: CUDNN_DEFAULT_MATH -> Tensor Core Operations are not
                     // used 1: CUDNN_TENSOR_OP_MATH -> The use of Tensor Core
                     // Operations is permitted.

  int dimA[] = {(int)x.dim.dims[0], (int)x.dim.dims[1], (int)x.dim.dims[2], (int)x.dim.dims[3]};  // N, C, H, W;
  int padA[] = {(int)params.padding, (int)params.padding};
  int convstrideA[] = {(int)params.stride, (int)params.stride};
  // batch size and feature layers must be multiples of 4 or 32 when using int8x4 or int8x32 respectively
  int filterdimA[] = {(int)w.dim.dims[0], (int)w.dim.dims[1], (int)w.dim.dims[2], (int)w.dim.dims[3]}; //k, c, r, s

  cudnnTensorFormat_t  filterFormat = CUDNN_TENSOR_NCHW;

#ifdef PRINT_VERBOSE
  PDBG("Testing using cudnn forward\n");
#endif
  status_t ret =
      doForward<T>(x, w, y, dimA, padA, convstrideA, filterdimA, filterFormat,
                   CUDNN_DATA_FLOAT, mathType,
                   handle_, cudnnIdesc, cudnnFdesc, cudnnOdesc, cudnnConvDesc);

  // shadow copy
  tensor_t cached_x_shadow = x;
  tensor_t cached_w_shadow = w;

  // TODO put w and data
  if (cache) {
    lcache_push(cache, cached_x_shadow);
    lcache_push(cache, cached_w_shadow);
  }
  return ret;
}

status_t convolution_backward_cudnn(tensor_t dx, tensor_t dw, lcache_t* cache,
                                    conv_param_t const params,
                                         tensor_t const dout) {
  tensor_t x, w;

  // NOTE : the order of pop matters, should be flattened_x, w, x (reverse of
  // forward)
  w = lcache_pop(cache);
  x = lcache_pop(cache);

  int mathType = 0;

  int dimA[] = {(int)dx.dim.dims[0], (int)dx.dim.dims[1], (int)dx.dim.dims[2], (int)dx.dim.dims[3]};  // N, C, H, W;
  int padA[] = {(int)params.padding, (int)params.padding};
  int convstrideA[] = {(int)params.stride, (int)params.stride};
  //batch size and feature layers must be multiples of 4 or 32 when using int8x4 or int8x32 respectively
  int filterdimA[] = {(int)w.dim.dims[0], (int)w.dim.dims[1],
                      (int)w.dim.dims[2], (int)w.dim.dims[3]};  // k, c, r, s

  cudnnTensorFormat_t  filterFormat = CUDNN_TENSOR_NCHW;

#ifdef PRINT_VERBOSE
  PDBG("Testing using cudnn backward data\n");
#endif

  status_t ret =
      doBackward<T>(x, dx, w, dw, dout, dimA, padA, convstrideA, filterdimA, filterFormat, CUDNN_DATA_FLOAT,
                               mathType);

  return ret;
}
