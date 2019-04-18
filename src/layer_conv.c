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
  for (int i = 0; i < sz; ++i) {
    y.data[i] = tpose.data[i];
  }
  y.mem_type = tpose.mem_type;

  // fill cache
  // NOTE, the order matters should be x, w, flattened_x
  if(cache) {
    lcache_push(cache, x);
    lcache_push(cache, w);
    lcache_push(cache, flattened_x);
  }

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
//status_t im2col_inner(tensor_t cols, tensor_t x_padded,
//                      uint N, uint C, uint H, uint W, uint HH, uint WW,
//                      uint filter_height, uint filter_width, uint padding, uint stride){
//
//  uint cols_d_1 = cols.dim.dims[1];
//  uint img_sz = C * x_padded.dim.dims[2] * x_padded.dim.dims[3];
//  uint chan_sz = x_padded.dim.dims[2] * x_padded.dim.dims[3];
//  uint row_sz = x_padded.dim.dims[2];
//
//  for (uint c = 0; c < C; c++) // for each channel
//    for (uint yy = 0; yy < HH; yy++) // stride over rows
//      for (uint xx = 0; xx < WW; xx++) // stride over cols
//        for (uint ii = 0; ii < filter_height; ii++) // for each row of filter
//          for (uint jj = 0; jj < filter_width; jj++){ // for each col of filter
//            uint row = c * filter_width * filter_height + ii * filter_height + jj;
//            for (uint i = 0; i < N; i++){
//              uint col = yy * WW * N + xx * N + i;
//              uint target_idx = row * cols_d_1 + col;

//              uint src_idx = (i * img_sz) + (c * chan_sz) + (stride * yy + ii) * row_sz + stride * xx + jj;
//              cols.data[target_idx] = x_padded.data[src_idx];
//            }
//          }
//
//  return S_OK;
//}
status_t im2col_inner(tensor_t cols, tensor_t x_padded,
                      uint N, uint C, uint H, uint W, uint HH, uint WW,
                      uint filter_height, uint filter_width, uint padding, uint stride){

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
              uint first_elem = window_index_r * stride * W + windows_index_c * stride;
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
  for (int i = 0; i < capacity; ++i) {
    dx.data[i] = t.data[i];
  }


  tensor_destroy(&dout_reshaped);
  tensor_destroy(&x_cols_T);
  tensor_destroy(&w_T);
  tensor_destroy(&t);

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
tensor_t col2im(tensor_t dx_cols, uint N, uint C, uint H, uint W, uint field_height, uint field_width, uint padding, uint stride)
{
  uint HH = (H + 2 * padding - field_height) / stride + 1;
  uint WW = (W + 2 * padding - field_width) / stride + 1;

  uint x_padded_shape[] = { N, C, H + 2 * padding, W + 2 * padding };
  tensor_t x_padded = tensor_make_scalar(x_padded_shape, ARRAY_SIZE(x_padded_shape), 0);                                    // new mem created by returned

  col2im_inner(dx_cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride);
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
//void col2im_inner(tensor_t dx_cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW,
//                  uint field_height, uint field_width, uint padding, uint stride)
//{
//  uint dx_col_d_1 = dx_cols.dim.dims[1];
//  uint x_p_d_1 = x_padded.dim.dims[1];
//  uint x_p_d_2 = x_padded.dim.dims[2];
//  uint x_p_d_3 = x_padded.dim.dims[3];
//
//
//  for (int c = 0; c < C; ++c) {
//    for (int ii = 0; ii < field_height; ++ii) {
//      for (int jj = 0; jj < field_width; ++jj) {
//        uint row = c * field_width * field_height + ii * field_width + jj;
//        for (int yy = 0; yy < HH; ++yy) {
//          for (int xx = 0; xx < WW; ++xx) {
//            for (int i = 0; i < N; ++i) {
//              uint col = yy * WW * N + xx * N + i;
//              uint src_idx = row * dx_col_d_1 + col;
//              uint target_idx =
//                  i * x_p_d_1 * x_p_d_2 * x_p_d_3
//                  + c * x_p_d_2 * x_p_d_3
//                  + (stride * yy + ii) * x_p_d_3
//                  + stride * xx + jj;
//              x_padded.data[target_idx] += dx_cols.data[src_idx];
//            }
//          }
//        }
//      }
//    }
//  }
//}


/*
 * this version attempts to set up the most basic version...
 * the idea is to collapse the first 4 dimensions into
 * 1, and then base the remaining dimensions off of the
 * collapsed dimensions.  This will allow us to parallelize
 * the algorithm in a very naive way.
 */
void col2im_inner(tensor_t dx_cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW,
                  uint field_height, uint field_width, uint padding, uint stride)
{
  uint dx_col_d1  = dx_cols.dim.dims[1];
  uint x_p_d1     = x_padded.dim.dims[1];
  uint x_p_d2     = x_padded.dim.dims[2];
  uint x_p_d3     = x_padded.dim.dims[3];

  printf("\n(N=%u, C=%u, H=%u, W=%u, HH=%u, WW=%u, field_h=%u, field_w=%u, p=%u, stride=%u)\n", N, C, H, W, HH, WW, field_height, field_width, padding, stride);

  uint iter = 0;
  uint outer_count = 0;
  for (uint i = 0; i < N; ++i) { // for each image
    for (uint c = 0; c < C; ++c) {  // for each channel
      for (uint fi = 0; fi < field_height; ++fi) {
        for (uint fj = 0; fj < field_width; ++fj) {
          outer_count++;
          uint row = c * field_width * field_height + fi * field_width + fj;

          uint ii = iter / (C * field_height * field_width);
          uint cc = (iter / (field_height * field_width)) % C;  // jj is the channel in the image
          uint fii = iter / (field_width) % field_height;
          uint fjj = iter % field_width;

          for (uint h = 0; h < HH; ++h) {
            for (uint w = 0; w < WW; ++w) {

              assert(ii == i);
              assert(cc == c);
              assert(fii == fi);
              assert(fjj == fj);

              printf("iter=%u, ii=%u, i=%u, cc=%u, c=%u, fii=%u, fi=%u, fjj=%u, fj=%u, h=%u, w=%u\n", iter, ii, i, cc, c, fii, fi, fjj, fj, h, w);
              uint col = h * WW * N + w * N + i;
              uint src_idx = row * dx_col_d1 + col;
              uint target_idx =
                  i * x_p_d1 * x_p_d2 * x_p_d3
                  + c * x_p_d2 * x_p_d3
                  + (stride * h + fi) * x_p_d3
                  + stride * w + fj;
//              printf("x_padded.data[%u]=%f added to dx_cols.data[%u]=%f --> ", target_idx, x_padded.data[target_idx], src_idx, dx_cols.data[src_idx]);
              x_padded.data[target_idx] += dx_cols.data[src_idx];
//              printf("%f\n", x_padded.data[target_idx]);
            }
          }
          ++iter;
        }
      }
    }
  }
  printf("outer count = %u\n", outer_count);
//  printf("after x_padded");
//  tensor_print_flat(x_padded);
}

/*
 * this version gets us closer to the upgraded version mapping
 * the entire iter space to the ii, cc, fii, fjj elements.
 *
 * It does not calculate the h and w from iter yet.
 */
//void col2im_inner(tensor_t dx_cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW,
//                  uint field_height, uint field_width, uint padding, uint stride)
//{
//  uint dx_col_d1  = dx_cols.dim.dims[1];
//  uint x_p_d1     = x_padded.dim.dims[1];
//  uint x_p_d2     = x_padded.dim.dims[2];
//  uint x_p_d3     = x_padded.dim.dims[3];
//
//  uint C_fh_fw_HH_WW = C * field_height * field_width * HH * WW;
//
//  printf("\n(N=%u, C=%u, H=%u, W=%u, HH=%u, WW=%u, field_h=%u, field_w=%u, p=%u, stride=%u)\n", N, C, H, W, HH, WW, field_height, field_width, padding, stride);
//
//  uint iter = 0;
//  uint outer_count = 0;
//  for (uint i = 0; i < N; ++i) { // for each image
//    for (uint c = 0; c < C; ++c) {  // for each channel
//      for (uint fi = 0; fi < field_height; ++fi) {
//        for (uint fj = 0; fj < field_width; ++fj) {
//          outer_count++;
//
//          for (uint h = 0; h < HH; ++h) {
//            for (uint w = 0; w < WW; ++w) {
//              uint ii = iter / C_fh_fw_HH_WW;  // ii is the target image
//              uint cc = (iter / (field_height * field_width * HH * WW)) % C;  // jj is the channel in the image
//              uint fii = iter / (HH * WW * field_width) % field_height;
//              uint fjj = (iter / (HH * WW)) % field_width;
//
//              uint row = c * field_width * field_height + fi * field_width + fj;
//
//              assert(ii == i);
//              assert(cc == c);
//              assert(fii == fi);
//              assert(fjj == fj);
//
//              printf("iter=%u, i=%u, c=%u, fi=%u, fj=%u, h=%u, w=%u\n", iter, i, c, fi, fj, h, w);
//              uint col = h * WW * N + w * N + i;
//              uint src_idx = row * dx_col_d1 + col;
//              uint target_idx =
//                  i * x_p_d1 * x_p_d2 * x_p_d3
//                  + c * x_p_d2 * x_p_d3
//                  + (stride * h + fi) * x_p_d3
//                  + stride * w + fj;
//              printf("x_padded.data[%u]=%f added to dx_cols.data[%u]=%f --> ", target_idx, x_padded.data[target_idx], src_idx, dx_cols.data[src_idx]);
//              x_padded.data[target_idx] += dx_cols.data[src_idx];
//              printf("%f\n", x_padded.data[target_idx]);
//              ++iter;
//            }
//          }
//        }
//      }
//    }
//  }
//  printf("outer count = %u\n", outer_count);
////  printf("after x_padded");
////  tensor_print_flat(x_padded);
//}