#include "awnn/layer_conv.h"

#include <printf.h>


status_t convolution_forward(tensor_t const x, tensor_t const w, lcache_t * cache, conv_param_t const params, tensor_t y){

  // 1. flatten the input into vectors which represent the filters
  tensor_t flattened_x = im2col(x, w, params);

//  tensor_dump(flattened_x);

  // 2. this is where the filters are actually applied
  uint const reshaped_w_shape[] = { w.dim.dims[0], w.dim.dims[1] * w.dim.dims[2] * w.dim.dims[3] };
  tensor_t reshaped_w = tensor_make_copy(w);
  tensor_reshape_(&reshaped_w, reshaped_w_shape, ARRAY_SIZE(reshaped_w_shape));

  uint const out_shape[] = { w.dim.dims[0], flattened_x.dim.dims[1] };
  tensor_t out = tensor_make(out_shape, ARRAY_SIZE(out_shape));

  tensor_matmul(reshaped_w, flattened_x, out);


  //  tensor_t res = w.reshape((w.shape[0], -1)).dot(x_cols);
//
//  //##### convert output back to appropriate shape
//  out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0]);
  tensor_dump(out);
  // want w[0]: 2 x out[2] : 0 x
  uint const out_shape_2[] = { w.dim.dims[0], y.dim.dims[2], y.dim.dims[3], x.dim.dims[0] };
  tensor_reshape_(&out, out_shape_2, ARRAY_SIZE(out_shape_2));
  tensor_dump(out);

  tensor_destroy(y);
  tensor_copy(out, y);

  //  // 3. transpose output
  //  out = out.transpose(3, 0, 1, 2);
  tensor_t tpose = tensor_transpose_3012(out);
//
//  // fill cache
//  cache = (x, w, conv_param, x_cols);

  return S_OK;
}


tensor_t im2col(tensor_t const x, tensor_t const w, conv_param_t const params) {
  uint N, C, H, W, filter_height, filter_width, stride, pad;
  N = x.dim.dims[0];
  C = x.dim.dims[1];
  H = x.dim.dims[2];
  W = x.dim.dims[3];

  filter_height   = w.dim.dims[2];
  filter_width    = w.dim.dims[3];

  stride  = params.stride;
  pad     = params.padding;

  // Check dimensions
  assert((W + 2 * pad - filter_width) % stride == 0);
  assert((H + 2 * pad - filter_height) % stride == 0);

  uint HH = (H + 2 * pad - filter_height) / stride + 1; // total strides needed over rows
  uint WW = (W + 2 * pad - filter_width) / stride + 1; // total strides needed over cols

  // TODO: Optimize tensor_make_padded function
  tensor_t x_padded = tensor_make_padded_square_input(x, pad, 0);

  uint cols_shape[] = {C * filter_height * filter_width, N * HH * WW};

  tensor_t cols = tensor_make_zeros(cols_shape, 2); // set ndims=2

  im2col_inner(cols, x_padded, N, C, H, W, HH, WW, filter_height, filter_width, pad, stride);

  return cols;
}


// note that this strides along columns of the target "cols" tensor
// possibly could be re-written to take advantage of
status_t im2col_inner(tensor_t cols, tensor_t x_padded,
                      uint N, uint C, uint H, uint W, uint HH, uint WW,
                      uint filter_height, uint filter_width, uint padding, uint stride){

  uint img_sz = C * x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint chan_sz = x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint row_sz = x_padded.dim.dims[2];

  for (uint c = 0; c < C; c++) // for each channel
    for (uint yy = 0; yy < HH; yy++) // stride over rows
      for (uint xx = 0; xx < WW; xx++) // stride over cols
        for (uint ii = 0; ii < filter_height; ii++) // for each row of filter
          for (uint jj = 0; jj < filter_width; jj++){ // for each col of filter
            uint row = c * filter_width * filter_height + ii * filter_height + jj;
            for (uint i = 0; i < N; i++){
              uint col = yy * WW * N + xx * N + i;
              uint target_idx = row * cols.dim.dims[1] + col;
              uint src_idx = (i * img_sz) + (c * chan_sz) + (stride * yy + ii) * row_sz + stride * xx + jj;
              cols.data[target_idx] = x_padded.data[src_idx];
            }
          }

  return S_OK;
}




status_t convolution_backward(tensor_t dx, tensor_t dw, lcache_t const *cache, tensor_t const dy){
  status_t ret = S_ERR;

  ret = S_OK;

  return ret;
}
