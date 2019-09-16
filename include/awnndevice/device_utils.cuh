//
// Created by cmgoebel on 4/25/19.
//

#pragma once

#include "range.cuh"
#include "awnn/common.h"
#include "awnn/tensor.h"

// type alias to simplify typing for step range...
using namespace util::lang;

template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template <typename T>
static __device__ step_range<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  return range(begin, end).step(gridDim.x * blockDim.x);
}

static __device__ int global_idx() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x
      + gridDim.x * gridDim.y * blockIdx.z;
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
      + (threadIdx.z * (blockDim.x * blockDim.y))
      + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

static __device__ uint d_capacity(tensor_t t) {
  uint c = 0;

  if (t.dim.dims[0] == 0) {
    return c;
  } else {
    c = t.dim.dims[0];
  }

  for (int i = 1; i < MAX_DIM; ++i) {
    if (t.dim.dims[i] == 0) {
      return c;
    }
    c *= t.dim.dims[i];
  }

  return c;
}

static __global__ void print_tensor_device(tensor_t t) {
  if (threadIdx.x == 0) {
    printf("entered print_tensor_device capacity = %u\n", d_capacity(t));

    printf("[");
    uint i = 0;
    for (; i < d_capacity(t) - 1; ++i) {
      printf("%f ", i, t.data[i]);
    }
    printf("%f]\n", t.data[i]);
  }
}

static __global__ void elementwise_add_inplace_device(tensor_t a, tensor_t const b) {
  if(global_idx() == 0) {
    assert(a.mem_type == GPU_MEM);
    assert(b.mem_type == GPU_MEM);

    assert(d_capacity(a) == d_capacity(b));
  }

  for (uint i : grid_stride_range(0u, d_capacity(a))) {
    a.data[i] += b.data[i];
  }
}

static __global__ void elementwise_mul_inplace_device(tensor_t a, tensor_t b) {
  if(global_idx() == 0) {
    assert(a.mem_type == GPU_MEM);
    assert(b.mem_type == GPU_MEM);

    assert(d_capacity(a) == d_capacity(b));
  }

  for (uint i : grid_stride_range(0u, d_capacity(a))) {
    a.data[i] *= b.data[i];
  }
}


static __global__ void build_mask_device(tensor_t x, tensor_t mask) {
  if(global_idx() == 0) {
    assert(x.mem_type == GPU_MEM);
    assert(mask.mem_type == GPU_MEM);

    assert(d_capacity(x) == d_capacity(mask));
  }

  for (uint i : grid_stride_range(0u, d_capacity(mask))) {
    mask.data[i] = x.data[i] > 0 ? 1.0 : 0.0;
  }
}

// write 0's into a device buffer
static __global__ void tensor_fill_scalar_device(tensor_t t, T scalar)
{
  assert(t.mem_type == GPU_MEM);
  for (uint i : grid_stride_range(0u, d_capacity(t))) {
    t.data[i] = 0;
  }
}

// write 0's into a device buffer
static __global__ void tensor_copy_d2d(tensor_t copy_to, tensor_t copy_from)
{
  if(global_idx() == 0) {
    assert(copy_to.mem_type == GPU_MEM);
    assert(copy_from.mem_type == GPU_MEM);

    assert(copy_to.dim.dims[0] == copy_from.dim.dims[0]);
    assert(copy_to.dim.dims[1] == copy_from.dim.dims[1]);
    assert(copy_to.dim.dims[2] == copy_from.dim.dims[2]);
    assert(copy_to.dim.dims[3] == copy_from.dim.dims[3]);
  }

  for (uint i : grid_stride_range(0u, d_capacity(copy_to))) {
    copy_to.data[i] = copy_from.data[i];
  }
}

// This function should only be required if  __CUDA_ARCH__ < 600
// currently I have it working at all times when the AWNN_USE_FLT32
// is not defined.
// TODO : remove the AWNN_USE_FLT32 and figure out how to enable this
// TODO : function with the preprocessor directly
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
//#if __CUDA_ARCH__ < 600
static __device__ double atomicAddDouble(double* address, double val)
{
  unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
//#endif