#include "awnn/layer_conv.h"


status_t convolution_forward(tensor_t const x, tensor_t const w, lcache_t * cache, conv_param_t const params, tensor_t y){
  status_t ret = S_ERR;

  // im2col with input

  // im2col with kernels

  // matmul

  // col2img

  ret = S_OK;

  return ret;
}


status_t convolution_backward(tensor_t dx, tensor_t dw, lcache_t const *cache, tensor_t const dy){
  status_t ret = S_ERR;

  ret = S_OK;

  return ret;
}
