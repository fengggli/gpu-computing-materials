#include "awnn/layer_conv.h"

#include <printf.h>


status_t convolution_forward(tensor_t const x, tensor_t const w, lcache_t * cache, conv_param_t const params, tensor_t y){

  // 1. flatten the input into vectors which represent the filters
  tensor_t flattened_x = im2col(x, w, params);

  // 2. setup and apply filters
  // TODO : const input is preventing reshape, but this memory doesn't need to be allocated
  //        w is just used as a multiplier, but it needs to be reshaped.
  uint const reshaped_w_shape[] = { w.dim.dims[0], w.dim.dims[1] * w.dim.dims[2] * w.dim.dims[3] };
  tensor_t reshaped_w = tensor_make_copy(w);
  tensor_reshape_(&reshaped_w, reshaped_w_shape, ARRAY_SIZE(reshaped_w_shape));

  uint const out_shape[] = { w.dim.dims[0], flattened_x.dim.dims[1] };
  tensor_t out = tensor_make(out_shape, ARRAY_SIZE(out_shape));

  // apply here !!!
  tensor_matmul(reshaped_w, flattened_x, out);

  uint const out_shape_2[] = { w.dim.dims[0], y.dim.dims[2], y.dim.dims[3], x.dim.dims[0] };
  tensor_reshape_(&out, out_shape_2, ARRAY_SIZE(out_shape_2));

  // 3. transpose output
  tensor_t tpose = tensor_make_transpose_3012(out);

  // copy transposed to y
  y.dim = tpose.dim;
  uint sz = dim_get_capacity(tpose.dim);
  for (uint i = 0; i < sz; ++i) {
    y.data[i] = tpose.data[i];
  }
  y.mem_type = tpose.mem_type;

  // fill cache
  // NOTE, the order matters should be x, w, flattened_x
  if(cache) {
    lcache_push(cache, x);
    lcache_push(cache, w);
    lcache_push(cache, flattened_x);
  } else {
    tensor_destroy(&flattened_x);
  }

  tensor_destroy(&reshaped_w);
  tensor_destroy(&tpose);
  tensor_destroy(&out);
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

  // TODO : Optimize tensor_make_padded function
  // TODO : look into not allocating here... maybe check bounds in the inner
  tensor_t x_padded = tensor_make_padded_square_input(x, pad, 0);

  uint cols_shape[] = {C * filter_height * filter_width, N * HH * WW};

  tensor_t cols = tensor_make_zeros(cols_shape, 2); // set ndims=2

  im2col_inner(cols, x_padded, N, C, H, W, HH, WW, filter_height, filter_width, pad, stride);

  tensor_destroy(&x_padded);

  return cols;
}


// note that this strides along columns of the target "cols" tensor
// possibly could be re-written to take advantage of
status_t im2col_inner(tensor_t cols, tensor_t x_padded,
                      uint N, uint C, uint H, uint W, uint HH, uint WW,
                      uint filter_height, uint filter_width, uint padding, uint stride){
  AWNN_NO_USE(H);
  AWNN_NO_USE(W);
  AWNN_NO_USE(padding);

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

/**
 * creates 4 new chunks of memory
 *  * dout_reshaped
 *  * x_cols_T
 *  * w_T
 *  * t : x_cols converted back to tensor form
 *
 * @param dx
 * @param dw
 * @param cache
 * @param conv_params
 * @param dout
 * @return
 */
status_t convolution_backward(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const conv_params, tensor_t const dout) {
  tensor_t x, w, x_cols;

  // NOTE : the order of pop matters, should be flattened_x, w, x (reverse of forward)
  x_cols = lcache_pop(cache);
  w = lcache_pop(cache);
  x = lcache_pop(cache);

  uint num_filters = w.dim.dims[0];
  uint w_channels = w.dim.dims[1];
  uint filter_height = w.dim.dims[2];
  uint filter_width = w.dim.dims[3];

  tensor_t dout_reshaped = tensor_make_transpose_1230(dout);
  uint dout_2d_shape[] = { num_filters, dout_reshaped.dim.dims[1] * dout_reshaped.dim.dims[2] * dout_reshaped.dim.dims[3] };
  tensor_reshape_(&dout_reshaped, dout_2d_shape, ARRAY_SIZE(dout_2d_shape));

  tensor_t x_cols_T = tensor_make_transpose(x_cols);

  uint mult_shape[] = { dout_reshaped.dim.dims[0], x_cols_T.dim.dims[1] };
  tensor_reshape_(&dw, mult_shape, ARRAY_SIZE(mult_shape));
  tensor_matmul(dout_reshaped, x_cols_T, dw);

  uint dw_shape[] = { num_filters, w_channels, filter_height, filter_width };
  tensor_reshape_(&dw, dw_shape, ARRAY_SIZE(dw_shape));

  // done getting dw

  // now get dx in column form multiplying the w_T with the d_out
  uint w_shape[] = { num_filters, w_channels * filter_height * filter_width };
  tensor_reshape_(&w, w_shape, ARRAY_SIZE(w_shape));
  tensor_t w_T = tensor_make_transpose(w);

  // next gotta get dx : first we get it in flat form,
  uint dx_cols_shape[] = { w_T.dim.dims[0], dout_reshaped.dim.dims[1] };
  tensor_t dx_cols = tensor_make(dx_cols_shape, ARRAY_SIZE(dx_cols_shape));
  tensor_matmul(w_T, dout_reshaped, dx_cols);

  // then we convert it back to tensor form
  tensor_t t = col2im(dx_cols, x.dim.dims[0], x.dim.dims[1], x.dim.dims[2], x.dim.dims[3], filter_height, filter_width, conv_params.padding, conv_params.stride);

  // copy date into dw (assumption is that dw is already correct shape)
  uint capacity = tensor_get_capacity(t);
  for (uint i = 0; i < capacity; ++i) {
    dx.data[i] = t.data[i];
  }


  tensor_destroy(&dout_reshaped);
  tensor_destroy(&x_cols_T);
  tensor_destroy(&w_T);
  tensor_destroy(&t);
  tensor_destroy(&x_cols);
  tensor_destroy(&dx_cols);

  return S_OK;
}

/*
# x = np.empty((N, C, H, W), dtype=cols.dtype)
  HH = int((H + 2 * padding - field_height) / stride + 1)
  WW = int((W + 2 * padding - field_width) / stride + 1)
  x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)

# Moving the inner loop to a C-function with no bounds checking improves
# performance quite a bit for col2im.
  col2im_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride)
  if padding > 0:
  return x_padded[:, :, padding:-padding, padding:-padding]
  return x_padded
 */
tensor_t col2im(tensor_t cols, uint N, uint C, uint H, uint W, uint field_height, uint field_width, uint padding, uint stride)
{
  uint HH = (H + 2 * padding - field_height) / stride + 1;
  uint WW = (W + 2 * padding - field_width) / stride + 1;

  uint x_padded_shape[] = { N, C, H + 2 * padding, W + 2 * padding };
  tensor_t x_padded = tensor_make_scalar(x_padded_shape, ARRAY_SIZE(x_padded_shape), 0);                                    // new mem created by returned

  col2im_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride);
  if (padding) {
    tensor_t padding_removed = tensor_make_remove_padding_square(x_padded, padding);
    tensor_destroy(&x_padded);
    return padding_removed;
  }
  return x_padded;
}


/*
    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride * yy + ii, stride * xx + jj] += cols[row, col]
 */
void col2im_inner(tensor_t cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW,
                  uint field_height, uint field_width, uint padding, uint stride)
{
  AWNN_NO_USE(H);
  AWNN_NO_USE(W);
  AWNN_NO_USE(padding);
  for (uint c = 0; c < C; ++c) {
    for (uint ii = 0; ii < field_height; ++ii) {
      for (uint jj = 0; jj < field_width; ++jj) {
        uint row = c * field_width * field_height + ii * field_height + jj;
        for (uint yy = 0; yy < HH; ++yy) {
          for (uint xx = 0; xx < WW; ++xx) {
            for (uint i = 0; i < N; ++i) {
              uint col = yy * WW * N + xx * N + i;
              uint src_idx = row * cols.dim.dims[1] + col;
              uint target_idx =
                  i * x_padded.dim.dims[1] * x_padded.dim.dims[2] * x_padded.dim.dims[3]
                  + c * x_padded.dim.dims[2] * x_padded.dim.dims[3]
                  + (stride * yy + ii) * x_padded.dim.dims[3]
                  + stride * xx + jj;
              x_padded.data[target_idx] += cols.data[src_idx];
            }
          }
        }
      }
    }
  }
}
