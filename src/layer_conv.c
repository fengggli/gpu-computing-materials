#include "awnn/layer_conv.h"
status_t im2col(tensor_t const x, tensor_t const w, lcache_t * cache, conv_param_t const params, tensor_t out) {
  uint N, C, H, W, num_filters, filter_height, filter_width, stride, pad, out_height, out_width;
  N = x.dim.dims[0];
  C = x.dim.dims[1];
  H = x.dim.dims[2];
  W = x.dim.dims[3];
  num_filters = w.dim.dims[0];
  filter_height = w.dim.dims[2];
  filter_width = w.dim.dims[3];

  stride = params.stride;
  pad = params.padding;

  // Check dimensions
  assert((W + 2 * pad - filter_width) % stride == 0);
  assert((H + 2 * pad -filter_height) % stride == 0);

  // Create output
  out_height = (H + 2 * pad - filter_height) / stride + 1
  out_width = (W + 2 * pad - filter_width) / stride + 1
  out = tensor_make_zeos(N, num_filters, out_height, out_width)

}

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
