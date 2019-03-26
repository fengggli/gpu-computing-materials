#include "awnn/layer_conv.h"

status_t im2col_inner(tensor_t cols, tensor_t x_padded,
                        uint N,  uint C,  uint H,  uint W,  uint HH, uint WW,
                        uint field_height, uint field_width, uint padding, uint stride){

  for (uint c = 0; c < C; c++)
    for (uint yy = 0; yy < HH; yy++)
      for (uint xx = 0; xx < WW; xx++)
        for (uint ii = 0; ii < field_height; ii++)
          for (uint jj = 0; jj < field_width; jj++){
            uint row = c * field_width * field_height + ii * field_height + jj;
            for (uint i = 0; i < N; i++){
              uint col = yy * WW * N + xx * N + i;
              cols.data[row * N + col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]; // Correct?
            }
          }

  return S_OK;
}

status_t im2col(tensor_t const x, tensor_t const w, lcache_t * cache, conv_param_t const params, tensor_t out) {
  uint N, C, H, W, num_filters, filter_height, filter_width, stride, pad, x_cols, res;
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
  assert((H + 2 * pad - filter_height) % stride == 0);

  uint HH = (H + 2 * pad - filter_height) / stride + 1;
  uint WW = (W + 2 * pad - filter_width) / stride + 1;

  // TODO: Optimize tensor_make_padded function
  tensor_t x_padded = tensor_make_padded_square_input(x, pad, 0);

  uint cols_shape[] = {C * filter_height * filter_width, N * HH * WW};

  tensor_t cols = tensor_make_zeros(cols_shape, 2); // set ndims=2

  im2col_inner(cols, x_padded, N, C, H, W, HH, WW, filter_height, filter_width, pad, stride);

  return S_OK;
}

status_t convolution_forward(tensor_t const x, tensor_t const w, lcache_t * cache, conv_param_t const params, tensor_t y){
  status_t ret = S_ERR;

  // 1. flatten the input into vectors which represent the filters
  ret = im2col(x, w, cache, params, y);

  // 2. this is where the filters are actually applied
  tensor_t res = w.reshape((w.shape[0], -1)).dot(x_cols);

  //##### convert output back to appropriate shape
  out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0]);

  // 3. transpose output
  out = out.transpose(3, 0, 1, 2);

  // fill cache
  cache = (x, w, conv_param, x_cols);

  return ret;
}


status_t convolution_backward(tensor_t dx, tensor_t dw, lcache_t const *cache, tensor_t const dy){
  status_t ret = S_ERR;

  ret = S_OK;

  return ret;
}
