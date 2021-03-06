#include "awnn/layer_conv.h"
#ifdef USE_NNPACK
#include "nnpack.h"
#include "pthreadpool.h"
#endif

#include <awnn/memory.h>
#include <printf.h>

#ifdef USE_OPENBLAS
#include "cblas.h"
#endif
#ifdef USE_MKL
#include "mkl.h"
#endif

conv_method_t g_conv_method = CONV_METHOD_NAIVE;

void set_conv_method(conv_method_t method) { g_conv_method = method; }
conv_method_t get_conv_method() { return g_conv_method; }

status_t convolution_forward_simple(tensor_t const x, tensor_t const w,
                                    lcache_t* cache, conv_param_t const params,
                                    tensor_t y);
status_t convolution_backward_simple(tensor_t dx, tensor_t dw, lcache_t* cache,
                                     conv_param_t const conv_params,
                                     tensor_t const dout);

status_t convolution_forward(tensor_t const x, tensor_t const w,
                             lcache_t* cache, conv_param_t const params,
                             tensor_t y) {
  status_t ret;
  switch (g_conv_method) {
#if defined(USE_NNPACK) && defined(AWNN_USE_FLT32)
    case CONV_METHOD_NNPACK_AUTO:
    case CONV_METHOD_NNPACK_ft8x8:
    case CONV_METHOD_NNPACK_ft16x16:
    case CONV_METHOD_NNPACK_wt8x8:
    case CONV_METHOD_NNPACK_REF:
    case CONV_METHOD_NNPACK_implicit_gemm:
    case CONV_METHOD_NNPACK_direct:
      ret = convolution_forward_nnpack(g_conv_method, x, w, cache, params, y);
      break;
#endif

    case CONV_METHOD_PERIMG:
      ret = conv_forward_perimg(x, w, cache, params, y);
      break;

    case CONV_METHOD_NAIVE:
    default:
      ret = convolution_forward_simple(x, w, cache, params, y);
      break;
  }
  return ret;
}

status_t convolution_backward(tensor_t dx, tensor_t dw, lcache_t* cache,
                              conv_param_t const conv_params,
                              tensor_t const dout) {
  status_t ret;
  switch (g_conv_method) {
#if defined(USE_NNPACK) && defined(AWNN_USE_FLT32)
    case CONV_METHOD_NNPACK_AUTO:
    case CONV_METHOD_NNPACK_ft8x8:
    case CONV_METHOD_NNPACK_ft16x16:
    case CONV_METHOD_NNPACK_wt8x8:
    case CONV_METHOD_NNPACK_implicit_gemm:
    case CONV_METHOD_NNPACK_direct:
    case CONV_METHOD_NNPACK_REF:
      ret = convolution_backward_nnpack(g_conv_method, dx, dw, cache,
                                        conv_params, dout);
      break;
#endif

    case CONV_METHOD_PERIMG:
      ret = conv_backward_perimg(dx, dw, cache, conv_params, dout);
      break;

    case CONV_METHOD_NAIVE:
    default:
      ret = convolution_backward_simple(dx, dw, cache, conv_params, dout);
      break;
  }
  return ret;
}

status_t convolution_forward_simple(tensor_t const x, tensor_t const w,
                                    lcache_t* cache, conv_param_t const params,
                                    tensor_t y) {
  // 1. flatten the input into vectors which represent the filters
  tensor_t flattened_x = im2col(x, w, params);  // NxCxHxW -> C*HH*WW x NxH'xW'

  // 2. setup and apply filters
  // TODO : const input is preventing reshape, but this memory doesn't need to
  // be allocated
  //        w is just used as a multiplier, but it needs to be reshaped.
  uint const reshaped_w_shape[] = {
      w.dim.dims[0],
      w.dim.dims[1] * w.dim.dims[2] * w.dim.dims[3]};  // (F, CxHHxWW)
  // TODO: feng this is not necessay. call gemm directly
  tensor_t reshaped_w = tensor_make_copy(w);
  tensor_reshape_(&reshaped_w, reshaped_w_shape, ARRAY_SIZE(reshaped_w_shape));

  uint const out_shape[] = {w.dim.dims[0],
                            flattened_x.dim.dims[1]};  // (F, NxH'xW')
  tensor_t out = tensor_make(out_shape, ARRAY_SIZE(out_shape));

  // apply here !!!
  tensor_matmul(reshaped_w, flattened_x, out);

  uint const out_shape_2[] = {w.dim.dims[0], y.dim.dims[2], y.dim.dims[3],
                              x.dim.dims[0]};  // F x H' x W' x N
  tensor_reshape_(&out, out_shape_2,
                  ARRAY_SIZE(out_shape_2));  // (F x N*H'*W') -> (F x H'xW'xN)

  // 3. transpose output
  tensor_t tpose = tensor_make_transpose_3012(out);  // Nx F x H' x W'

  // copy transposed to y
  y.dim = tpose.dim;
  uint sz = dim_get_capacity(tpose.dim);
  for (uint i = 0; i < sz; ++i) {
    y.data[i] = tpose.data[i];
  }
  y.mem_type = tpose.mem_type;

  // fill cache
  // NOTE, the order matters should be x, w, flattened_x
  if (cache) {
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
  uint N, C, H, W, filter_height, filter_width;
  int stride, pad;

  N = x.dim.dims[0];
  C = x.dim.dims[1];
  H = x.dim.dims[2];
  W = x.dim.dims[3];

  filter_height = w.dim.dims[2];
  filter_width = w.dim.dims[3];

  stride = params.stride;
  pad = params.padding;

  assert((W + 2 * pad - filter_width) % stride == 0);
  assert((H + 2 * pad - filter_height) % stride == 0);

  uint HH = (H + 2 * pad - filter_height) / stride + 1; // total strides needed over rows
  uint WW = (W + 2 * pad - filter_width) / stride + 1;  // total strides needed over cols

  // TODO : Optimize tensor_make_padded function
  // TODO : look into not allocating here... maybe check bounds in the inner
  tensor_t x_padded = tensor_make_padded_square_input(x, pad, 0);

  uint flattened_x_shape[] = {C * filter_height * filter_width, N * HH * WW};

  tensor_t flattened_x = tensor_make_zeros(flattened_x_shape, ARRAY_SIZE(flattened_x_shape));

  im2col_inner(flattened_x, x_padded, N, C, H, W, HH, WW, filter_height, filter_width, pad, stride);

  tensor_destroy(&x_padded);

  return flattened_x;
}

/**
 * The primary purpose of this function is to take the padded tensor, which potentially
 * has many dimensions, and convert it into the flattened view.  The flattened "cols" tensor
 * is arranged such that the filters are laid out along each row.
 *
 * So a single row represents the data in the original input that a single filter would touch.
 * This is true for each channel as well, and for each image.
 *
 * Multiple channels will extend the number of rows, with each channel grouped into a block
 * of rows, and then the next channel as the next block.  If there are multiple images, this
 * pattern will be repeated.
 *
 * In the GPU, this function could likely use shared memory because elements are *** sometimes ***
 * accessed repeatedly due to the fact that the filters frequently overlap (depending on the stride
 * and filter size).
 *
 * In cases where the data does not overlap, shared memory usage would actually result in a slower
 * kernel due to an extra pair of copies into shmem and back to global, but this could be a point
 * of optimization... if there is a big overlap, the reuse of elements could make shared mem
 * usage worth while.
 */
// note that this strides along columns of the target "cols" tensor
// possibly could be re-written to take advantage of
status_t im2col_inner(tensor_t cols, tensor_t x_padded,
                      uint N, uint C, uint H, uint W, uint HH, uint WW,
                      uint filter_height, uint filter_width, int padding, int stride){
  AWNN_NO_USE(H);
  AWNN_NO_USE(W);
  AWNN_NO_USE(padding);

  uint cols_d_1 = cols.dim.dims[1];
  uint img_sz = C * x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint chan_sz = x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint row_sz = x_padded.dim.dims[2];

  for (uint c = 0; c < C; c++)                       // for each channel
    for (uint yy = 0; yy < HH; yy++)                 // stride over rows
      for (uint xx = 0; xx < WW; xx++)               // stride over cols
        for (uint ii = 0; ii < filter_height; ii++)  // for each row of filter
          for (uint jj = 0; jj < filter_width;
               jj++) {  // for each col of filter
            uint row =
                c * filter_width * filter_height + ii * filter_height + jj;
            for (uint i = 0; i < N; i++) {
              uint col = yy * WW * N + xx * N + i;
              uint target_idx = row * cols.dim.dims[1] + col;
              uint src_idx = (i * img_sz) + (c * chan_sz) +
                             (stride * yy + ii) * row_sz + stride * xx + jj;
              cols.data[target_idx] = x_padded.data[src_idx];
            }
          }

  return S_OK;
}

#if 0
status_t im2col_inner(tensor_t cols, tensor_t x_padded,
                      uint N, uint C, uint H, uint W, uint HH, uint WW,
                      uint filter_height, uint filter_width, int padding, int stride){
  AWNN_NO_USE(H);
  AWNN_NO_USE(W);
  AWNN_NO_USE(padding);

  uint cols_d_1 = cols.dim.dims[1];
  uint img_sz = C * x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint chan_sz = x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint row_sz = x_padded.dim.dims[2];

  uint new_img_sz = x_padded.dim.dims[0] * x_padded.dim.dims[1] * x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint channel_sz = x_padded.dim.dims[2] * x_padded.dim.dims[3];

  uint filter_size = filter_height * filter_width;

  uint filters_per_channel = HH * WW;
  uint filters_per_image = C * filters_per_channel;
  uint total_filters = N * filters_per_image;

  // TODO deal with last week's time sheet
  uint iter = 0;
  for (uint n = 0; n < N; n++){
    for (uint c = 0; c < C; c++){ // for each channel
      for (uint j = 0; j < HH; j++) {  // total strides needed over rows
        for (uint k = 0; k < WW; k++) {  // total strides needed over cols

          for (uint f_row = 0; f_row < filter_height; ++f_row) {  // for each row of filter (relative row)
            for (uint f_col = 0; f_col < filter_width; ++f_col) {  // for each col of filter

              uint nn = iter / (filters_per_image * filter_size);  // nn is the target image
              uint cc = (iter / (filters_per_channel * filter_size)) % C;  // cc is the channel of the target filter
//              uint jj = (iter / WW) % HH; // jj is the target filter row
//              uint kk = (iter % WW);

              // TODO delete these unused elements
              uint t_row = iter / filter_size;
              uint t_col = iter % filter_size;
              uint t_idx = t_row * filter_size + t_col; // t_idx target index
              assert(t_idx == iter);

              // locate the window
              uint window_index_linear = iter / filter_size;
              uint window_index_r  = (window_index_linear / HH) % WW;
              uint windows_index_c = window_index_linear % WW;

              assert(nn == n);
              assert(cc == c);
              assert(window_index_r == j);
              assert(windows_index_c == k);

              // index of the first elem
              uint ff_row = (iter / filter_width) % filter_width;
              assert(ff_row == f_row);
              uint ff_col = iter % filter_width;
              assert(ff_col == f_col);

              uint row = c * filter_width * filter_height + f_row * filter_width + f_col;
              uint col = j * WW * N + k * N + n;
              uint target_idx = row * cols_d_1 + col;
              uint src_idx = (n * img_sz) + (c * chan_sz) + (stride * j + f_row) * row_sz + stride * k + f_col;
              cols.data[target_idx] = x_padded.data[src_idx];
//              printf("n=%u, c=%u, j=%u, k=%u, window_index_r=%u, windows_index_c=%u, window_idx_linear=%u, f_row=%u, f_col=%u, first_elem=%u, t_row=%u, t_col=%u, t_idx=%u, target_idx=%u, src_idx=%u, val=%f, row=%u, col=%u\n", n, c, j, k, window_index_r, windows_index_c, window_index_linear, f_row, f_col, first_elem, t_row, t_col, t_idx, target_idx, src_idx, cols.data[target_idx], row, col);
              ++iter;
            }
          }
        }
      }
    }
  }

  return S_OK;
}
#endif

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

status_t convolution_backward_simple(tensor_t dx, tensor_t dw, lcache_t* cache,
                                     conv_param_t const conv_params,
                                     tensor_t const dout) {
  tensor_t x, w, x_cols;

  // NOTE : the order of pop matters, should be flattened_x, w, x (reverse of
  // forward)
  x_cols = lcache_pop(cache);
  w = lcache_pop(cache);
  x = lcache_pop(cache);

  uint num_filters = w.dim.dims[0];
  uint w_channels = w.dim.dims[1];
  uint filter_height = w.dim.dims[2];
  uint filter_width = w.dim.dims[3];

  // 1. tensor transpose 1230 the dout (derivative of output layer)
  tensor_t dout_T_1230 = tensor_make_transpose_1230(dout);

  // 2. reshape the dout_T to a 2D shape by collapsing the last 3 dims
  uint dout_2d_shape[] = { num_filters, dout_T_1230.dim.dims[1] * dout_T_1230.dim.dims[2] * dout_T_1230.dim.dims[3] };
  tensor_reshape_(&dout_T_1230, dout_2d_shape, ARRAY_SIZE(dout_2d_shape));

  // 3. 2D transpose the flattened_x (x_cols)
  tensor_t x_cols_T = tensor_make_transpose(x_cols);

  // 4. multiply the transposed dout by the flat x_cols_T
  uint mult_shape[] = { dout_T_1230.dim.dims[0], x_cols_T.dim.dims[1] };
  tensor_reshape_(&dw, mult_shape, ARRAY_SIZE(mult_shape));
  tensor_matmul(dout_T_1230, x_cols_T, dw);

  // 5. reshape dw to same shape as cached w
  uint dw_shape[] = { num_filters, w_channels, filter_height, filter_width };

  tensor_reshape_(&dw, dw_shape, ARRAY_SIZE(dw_shape));

  // done getting dw (derivative of w)

  // 6. now get dx in column form multiplying the w_T with the d_out
  uint w_shape[] = { num_filters, w_channels * filter_height * filter_width };
  tensor_reshape_(&w, w_shape, ARRAY_SIZE(w_shape));
  tensor_t w_T = tensor_make_transpose(w);

  // next gotta get dx : first we get it in flat form,
  uint dx_cols_shape[] = { w_T.dim.dims[0], dout_T_1230.dim.dims[1] };
  tensor_t dx_cols = tensor_make(dx_cols_shape, ARRAY_SIZE(dx_cols_shape));
  tensor_matmul(w_T, dout_T_1230, dx_cols);

  // then we convert it back to tensor form
  tensor_t t = col2im(dx_cols, x.dim.dims[0], x.dim.dims[1], x.dim.dims[2],
                      x.dim.dims[3], filter_height, filter_width,
                      conv_params.padding, conv_params.stride);

  // copy data into dx (assumption is that dx is already correct shape)
  uint capacity = tensor_get_capacity(t);
  for (uint i = 0; i < capacity; ++i) {
    dx.data[i] = t.data[i];
  }

  tensor_destroy(&dout_T_1230);
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
  x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
dtype=cols.dtype)

# Moving the inner loop to a C-function with no bounds checking improves
# performance quite a bit for col2im.
  col2im_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width,
padding, stride) if padding > 0: return x_padded[:, :, padding:-padding,
padding:-padding] return x_padded
 */
tensor_t col2im(tensor_t cols, uint N, uint C, uint H, uint W,
                uint field_height, uint field_width, int padding,
                int stride) {
  uint HH = (H + 2 * padding - field_height) / stride + 1;
  uint WW = (W + 2 * padding - field_width) / stride + 1;

  uint x_padded_shape[] = {N, C, H + 2 * padding, W + 2 * padding};
  // TODO(Feng): move padding inside col2img
  tensor_t x_padded =
      tensor_make_scalar(x_padded_shape, ARRAY_SIZE(x_padded_shape),
                         0);  // new mem created by returned

  col2im_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width,
               padding, stride);
  if (padding) {
    tensor_t padding_removed =
        tensor_make_remove_padding_square(x_padded, padding);
    tensor_destroy(&x_padded);
    return padding_removed;
  }
  return x_padded;
}

void col2im_inner(tensor_t dx_cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW,
                  uint field_height, uint field_width, int padding, int stride)
{
  uint dx_col_d_1 = dx_cols.dim.dims[1];
  uint x_p_d_1 = x_padded.dim.dims[1];
  uint x_p_d_2 = x_padded.dim.dims[2];
  uint x_p_d_3 = x_padded.dim.dims[3];


  for (int c = 0; c < C; ++c) {
    for (int ii = 0; ii < field_height; ++ii) {
      for (int jj = 0; jj < field_width; ++jj) {
        uint row = c * field_width * field_height + ii * field_width + jj;
        for (int yy = 0; yy < HH; ++yy) {
          for (int xx = 0; xx < WW; ++xx) {
            for (int i = 0; i < N; ++i) {
              uint col = yy * WW * N + xx * N + i;
              uint src_idx = row * dx_col_d_1 + col;
              uint target_idx =
                  i * x_p_d_1 * x_p_d_2 * x_p_d_3
                  + c * x_p_d_2 * x_p_d_3
                  + (stride * yy + ii) * x_p_d_3
                  + stride * xx + jj;
              x_padded.data[target_idx] += dx_cols.data[src_idx];
            }
          }
        }
      }
    }
  }
}
