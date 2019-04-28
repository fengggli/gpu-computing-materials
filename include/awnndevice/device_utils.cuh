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
    printf("entered print_tensor_device capacity t= %u\n", threadIdx.x,
    d_capacity(t));

    for (int i = 0; i < d_capacity(t); ++i) {
      printf("data[%u] = %f ", i, t.data[i]);
    }
  }
  printf("\n");
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