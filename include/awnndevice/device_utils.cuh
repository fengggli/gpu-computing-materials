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

static uint __device__ d_capacity(tensor_t t) {
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