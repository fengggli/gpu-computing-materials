//
// Created by cmgoebel on 4/25/19.
//

#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"
#include "awnn/layer.h"   // for lcache_t

#include <cublas_v2.h> // for cublasHandle_t

#ifdef __cplusplus
extern "C" {
#endif

int set_blocks(int x);
int set_threads(int x);

status_t apply_activation_function(tensor_t t);

status_t convolution_forward_device(cublasHandle_t handle, tensor_t const d_x, tensor_t d_w, lcache_t* cache, conv_param_t const params, tensor_t d_y);

status_t convolution_forward_device_host_harness(cublasHandle_t handle,
                                                 tensor_t h_x, tensor_t h_w,
                                                 lcache_t* hcache,
                                                 conv_param_t hparams,
                                                 tensor_t h_y);


tensor_t im2col_device(tensor_t const d_x, tensor_t const d_w, conv_param_t const params);

status_t im2col_inner_device(tensor_t cols, tensor_t x_padded,
                             uint N,  uint C,  uint H,  uint W,  uint HH, uint WW,
                             uint filter_height, uint filter_width, uint padding, uint stride);

tensor_t tensor_make_padded_square_input_device(tensor_t h_t, uint p, T val);

tensor_t tensor_make_transpose_3012_device(tensor_t t);

status_t convolution_backward_device(cublasHandle_t handle, tensor_t d_dx, tensor_t d_dw, lcache_t* cache, conv_param_t const params, tensor_t const d_dout);
status_t convolution_backward_device_host_harness(cublasHandle_t handle,
    tensor_t h_dx, tensor_t h_dw, lcache_t* hcache, conv_param_t const params,
    tensor_t const h_dout);


tensor_t col2im_device(tensor_t cols,
                       uint N, uint C, uint H, uint W,
                       uint field_height, uint field_width, uint pad_sz, uint stride);


void col2im_inner_device(tensor_t cols, tensor_t x_padded,
                         uint N, uint C, uint H, uint W, uint HH, uint WW,
                         uint field_height, uint field_width, uint padding, uint stride);

tensor_t tensor_make_remove_padding_square_device(tensor_t t, uint p);

tensor_t tensor_make_transpose_1230_device(tensor_t t);

void elementwise_add_device_host_harness(tensor_t h_a, tensor_t h_b);
void elementwise_mul_device_host_harness(tensor_t h_a, tensor_t h_b);
void build_mask_device_host_harness(tensor_t h_a, tensor_t h_mask);

#ifdef __cplusplus
}
#endif
