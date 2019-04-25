//
// Created by cmgoebel on 4/25/19.
//

#pragma once

status_t convolution_forward_device(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y);


status_t convolution_forward_device_harness(tensor_t hx, tensor_t hw, lcache_t * hcache, conv_param_t hparams, tensor_t hy);

tensor_t matrix_dot_cublas_harness(tensor_t hx, tensor_t hy);

void cublasDot(const T * srcA, const T * srcB, T * out, int rowA, int colA, int colB);


tensor_t im2col_device(tensor_t const x, tensor_t const w, conv_param_t const params);


status_t im2col_inner_device(tensor_t cols, tensor_t x_padded,
                             uint N,  uint C,  uint H,  uint W,  uint HH, uint WW,
                             uint filter_height, uint filter_width, int padding, int stride);

tensor_t tensor_make_padded_square_input_device(tensor_t t, uint p, T val);

tensor_t tensor_make_transpose_3012_device(tensor_t t);

status_t convolution_backward_device(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout);

tensor_t col2im_device(tensor_t cols,
                       uint N, uint C, uint H, uint W,
                       uint field_height, uint field_width, int padding, int stride);


void col2im_inner_device(tensor_t cols, tensor_t x_padded,
                         uint N, uint C, uint H, uint W, uint HH, uint WW,
                         uint field_height, uint field_width, int padding, int stride);

tensor_t tensor_make_remove_padding_square_device(tensor_t t, uint p);

tensor_t tensor_make_transpose_1230_device(tensor_t t);
