//
// Created by Christopher Goebel on 2019-05-03.
//

#include "awnndevice/device_utils.cuh"
#include "awnndevice/device_utils_harness.cuh"

static int _elementwise_add_blocks{ 1 };
static int _elementwise_add_threads{ 1 };
static int _elementwise_mul_blocks{ 1 };
static int _elementwise_mul_threads{ 1 };
static int _build_mask_blocks{ 1 };
static int _build_mask_threads{ 1 };


int set_elementwise_add_blocks(int x) {
  _elementwise_add_blocks = x;
  return _elementwise_add_blocks;
}
int set_elementwise_add_threads(int x) {
  _elementwise_add_threads = x;
  return _elementwise_add_threads;
}
int set_elementwise_mul_blocks(int x) {
  _elementwise_mul_blocks = x;
  return _elementwise_mul_blocks;
}
int set_elementwise_mul_threads(int x) {
  _elementwise_mul_threads = x;
  return _elementwise_mul_threads;
}
int set_build_mask_blocks(int x) {
  _build_mask_blocks = x;
  return _build_mask_blocks;
}
int set_build_mask_threads(int x) {
  _build_mask_threads = x;
  return _build_mask_threads;
}

void elementwise_add_device_harness(tensor_t d_a, tensor_t d_b) {
  assert(d_a.mem_type == GPU_MEM);
  assert(d_b.mem_type == GPU_MEM);

  elementwise_add_inplace_device<<<_elementwise_add_blocks, _elementwise_add_threads>>>(d_a, d_b);
}

void elementwise_mul_device_harness(tensor_t d_a, tensor_t d_b) {
  assert(d_a.mem_type == GPU_MEM);
  assert(d_b.mem_type == GPU_MEM);

  elementwise_mul_inplace_device<<<_elementwise_mul_blocks, _elementwise_mul_threads>>>(d_a, d_b);
}

void build_mask_device_harness(tensor_t d_a, tensor_t d_mask) {
  assert(d_a.mem_type == GPU_MEM);
  assert(d_mask.mem_type == GPU_MEM);
  build_mask_device<<<_build_mask_blocks, _build_mask_threads>>>(d_a, d_mask);
}

void elementwise_add_device_host_harness(tensor_t h_a, tensor_t h_b) {
  assert(h_a.mem_type == CPU_MEM);
  assert(h_b.mem_type == CPU_MEM);

  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  elementwise_add_inplace_device<<<_elementwise_add_blocks, _elementwise_add_threads>>>(d_a, d_b);
  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}


void elementwise_mul_device_host_harness(tensor_t h_a, tensor_t h_b) {
  assert(h_a.mem_type == CPU_MEM);
  assert(h_b.mem_type == CPU_MEM);

  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  elementwise_mul_inplace_device<<<_elementwise_mul_blocks, _elementwise_mul_threads>>>(d_a, d_b);
  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}


void build_mask_device_host_harness(tensor_t h_a, tensor_t h_mask) {
  assert(h_a.mem_type == CPU_MEM);
  assert(h_mask.mem_type == CPU_MEM);

  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_mask = tensor_make_copy_h2d(h_mask);

  build_mask_device<<<_build_mask_blocks, _build_mask_threads>>>(d_a, d_mask);
  tensor_copy_d2h(h_mask, d_mask);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_mask);
}