//
// Created by Christopher Goebel on 2019-05-03.
//

#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void elementwise_add_device_host_harness(tensor_t h_a, tensor_t h_b);
void elementwise_mul_device_host_harness(tensor_t h_a, tensor_t h_b);
void build_mask_device_host_harness(tensor_t h_a, tensor_t h_mask);

#ifdef __cplusplus
}
#endif