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

int set_all_blocks(int x);
int set_all_threads(int x);

int set_copy_d2d_blocks(int x);
int set_copy_d2d_threads(int x);
int set_im2col_inner_blocks(int x);
int set_im2col_inner_threads(int x);
int set_col2im_inner_blocks(int x);
int set_col2im_inner_threads(int x);
int set_make_padded_blocks(int x);
int set_make_padded_threads(int x);
int set_remove_padding_blocks(int x);
int set_remove_padding_threads(int x);
int set_transpose_3012_blocks(int x);
int set_transpose_3012_threads(int x);
int set_transpose_1230_blocks(int x);
int set_transpose_1230_threads(int x);

status_t convolution_forward_device(cublasHandle_t handle, tensor_t const d_x, tensor_t d_w, lcache_t* cache, conv_param_t const params, tensor_t d_y);

status_t convolution_forward_device_host_harness(cublasHandle_t handle,
                                                 tensor_t h_x, tensor_t h_w,
                                                 lcache_t* hcache,
                                                 conv_param_t hparams,
                                                 tensor_t h_y);


tensor_t im2col_device(tensor_t const d_x, tensor_t const d_w, conv_param_t const params);

status_t im2col_inner_device(tensor_t cols, tensor_t x_padded,
                             int N,  int C,  int H,  int W,  int HH, int WW,
                             int filter_height, int filter_width, int padding, int stride);

tensor_t tensor_make_padded_square_input_device(tensor_t h_t, int p, T val);

tensor_t tensor_make_transpose_3012_device(tensor_t t);

status_t convolution_backward_device(cublasHandle_t handle, tensor_t d_dx, tensor_t d_dw, lcache_t* cache, conv_param_t const params, tensor_t const d_dout);
status_t convolution_backward_device_host_harness(cublasHandle_t handle,
    tensor_t h_dx, tensor_t h_dw, lcache_t* hcache, conv_param_t const params,
    tensor_t const h_dout);


tensor_t col2im_device(tensor_t cols,
                       int N, int C, int H, int W,
                       int field_height, int field_width, int pad_sz, int stride);


void col2im_inner_device(tensor_t cols, tensor_t x_padded,
                         int N, int C, int H, int W, int HH, int WW,
                         int field_height, int field_width, int padding, int stride);

tensor_t tensor_make_remove_padding_square_device(tensor_t t, int p);

tensor_t tensor_make_transpose_1230_device(tensor_t t);



#ifdef __cplusplus
}
#endif
