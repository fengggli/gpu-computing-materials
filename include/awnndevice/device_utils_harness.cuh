//
// Created by Christopher Goebel on 2019-05-03.
//

#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int set_elementwise_add_blocks(int x);
int set_elementwise_add_threads(int x);
int set_elementwise_mul_blocks(int x);
int set_elementwise_mul_threads(int x);
int set_build_mask_blocks(int x);
int set_build_mask_threads(int x);

void elementwise_add_device_harness(tensor_t d_a, tensor_t d_b);
void elementwise_mul_device_harness(tensor_t d_a, tensor_t d_b);
void build_mask_device_harness(tensor_t d_a, tensor_t d_mask);

void elementwise_add_device_host_harness(tensor_t h_a, tensor_t h_b);
void elementwise_mul_device_host_harness(tensor_t h_a, tensor_t h_b);
void build_mask_device_host_harness(tensor_t h_a, tensor_t h_mask);

#ifdef __cplusplus
}
#endif