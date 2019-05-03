//
// Created by Christopher Goebel on 2019-05-03.
//

#include "awnndevice/device_utils_host_harness.cuh"

void elementwise_add_device_host_harness(tensor_t h_a, tensor_t h_b) {
  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  elementwise_add_inplace_device<<<_blocks, _threads>>>(d_a, d_b);
  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}


void elementwise_mul_device_host_harness(tensor_t h_a, tensor_t h_b) {
  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  elementwise_mul_inplace_device<<<_blocks, _threads>>>(d_a, d_b);
  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}


void build_mask_device_host_harness(tensor_t h_a, tensor_t h_mask) {
  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_mask = tensor_make_copy_h2d(h_mask);

  build_mask_device<<<_blocks, _threads>>>(d_a, d_mask);
  tensor_copy_d2h(h_mask, d_mask);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_mask);
}