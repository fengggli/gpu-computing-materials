/*
 * Description: cudnn layer header
 *
 * Author: Yuankun Fu
 * e-mail: fu121@purdue.edu
 */

#include <cuda_runtime.h>
#include <cudnn.h>

#define checkCudaErr(...)       do { int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); if (err) goto clean; } while (0)

#define checkCudnnErr(...)      do { int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  if (err) goto clean; } while (0)

static int checkCudaError(cudaError_t code, const char* expr, const char* file, int line);
static int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line);

status_t convolution_forward_cudnn(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y,
                                   cudnnHandle_t handle_, cudnnTensorDescriptor_t cudnnIdesc,
                                   cudnnFilterDescriptor_t cudnnFdesc,
                                   cudnnTensorDescriptor_t cudnnOdesc,
                                   cudnnConvolutionDescriptor_t cudnnConvDesc);

status_t convolution_backward_cudnn(tensor_t dx, tensor_t dw, lcache_t* cache,
                                    conv_param_t const params, tensor_t const dout,
                                    cudnnHandle_t handle_, cudnnTensorDescriptor_t cudnnIdesc,
                                    cudnnFilterDescriptor_t cudnnFdesc,
                                    cudnnTensorDescriptor_t cudnnOdesc,
                                    cudnnConvolutionDescriptor_t cudnnConvDesc);

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