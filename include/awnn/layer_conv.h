/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/layer.h"
#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum conv_method_ {
  CONV_METHOD_NNPACK_AUTO = 0,
  CONV_METHOD_NNPACK_ft8x8 = 1,
  CONV_METHOD_NNPACK_ft16x16 = 2,
  CONV_METHOD_NNPACK_wt8x8 = 3,
  CONV_METHOD_NNPACK_implicit_gemm = 4,
  CONV_METHOD_NNPACK_direct = 5,
  CONV_METHOD_NNPACK_REF = 6,
  CONV_METHOD_NAIVE = 7,  // This is our convolution method

} conv_method_t;

void set_conv_method(conv_method_t);

/*
 * @brief forwarding for conv2d
 *
 * @param x data input of shape (N, C, H, W)
 * @param w filters, shape (F, C, HH, WW)
 * @param cache [output] intermidate results
 * @param y [output] forwarding output, shape (N,F,H',W'), H' = 1 + (H + 2 * pad
 * - HH) / stride
 *
 * @return S_OK if success, otherwise S_ERR or define your error type in
 * common.h
 *
 * Note:
 *  * all tensor_t should be pre-allocated
 *  * cache will be populated by forward function.
 *
 * See conv_forward_naive
 * https://github.com/fengggli/cs231n-assignments/blob/master/assignment2/cs231n/layers.py
 */

status_t convolution_forward(tensor_t const x, tensor_t const w,
                             lcache_t* cache, conv_param_t const params,
                             tensor_t y);

tensor_t im2col(tensor_t const x, tensor_t const w, conv_param_t const params);

status_t im2col_inner(tensor_t cols, tensor_t x_padded, int N, int C, int H,
                      int W, int HH, int WW, int filter_height,
                      int filter_width, int padding, int stride);

/*
 * @brief backprop
 *
 * @param dx [output] gradient w.r.t input
 * @param dw [output] gradient w.r.t filters
 * @param cache
 * @param dy gradient from upper layer
 *
 * @return S_OK if success, otherwise S_ERR or define your error type in
 * common.h
 */
status_t convolution_backward(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout);


tensor_t col2im(tensor_t dx_cols,
                int N, int C, int H, int W,
                int field_height, int field_width, int padding, int stride);


void col2im_inner(tensor_t dx_cols, tensor_t x_padded,
                  int N, int C, int H, int W, int HH, int WW,
                  int field_height, int field_width, int padding, int stride);


#ifdef USE_NNPACK
status_t convolution_forward_nnpack(conv_method_t, tensor_t const x,
                                    tensor_t const w, lcache_t* cache,
                                    conv_param_t const params, tensor_t y);

status_t convolution_backward_nnpack(conv_method_t, tensor_t dx, tensor_t dw,
                                     lcache_t* cache, conv_param_t const params,
                                     tensor_t const dout);
#endif

#ifdef __cplusplus
}
#endif
