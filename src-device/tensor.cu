#include "awnn/tensor.h"
#include "awnndevice/device_utils.cuh"

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
#include "awnn/memory.h"
#endif


#include <cuda_runtime_api.h>  // cudaMemset

void* mem_alloc_device(size_t size) {
  T* d_data;
  cudaError_t cudaStat;
  cudaStat = cudaMalloc(&d_data, size);
  AWNN_CHECK_EQ(cudaStat, cudaSuccess);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
  INC_TOTAL_TENSOR_ALLOC_DEVICE();
#endif
  return d_data;
}

void mem_free_device(void* d_data) {
  if (d_data) {
    cudaFree(d_data);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
    INC_TOTAL_TENSOR_DEALLOC_DEVICE();
#endif
  }
}

tensor_t tensor_make_device(uint const shape[], uint const ndims) {
  tensor_t t_device;

  uint capacity = 1;
  uint i;
  for (i = 0; i < ndims; ++i) {
    capacity *= shape[i];
    t_device.dim.dims[i] = shape[i];
  }
  for (; i < MAX_DIM; ++i) {
    t_device.dim.dims[i] = 0;
  }

  T* d_data = (T*)mem_alloc_device(capacity * sizeof(T));  // raw data at gpu mem in flat format

  t_device.data = d_data;
  t_device.mem_type = GPU_MEM;

  return t_device;
}

tensor_t tensor_make_zeros_device(uint const shape[], uint const ndims) {
  tensor_t t = tensor_make_device(shape, ndims);
  cudaMemset(t.data, 0, tensor_get_capacity(t) * sizeof(T));

  return t;
}

tensor_t tensor_make_copy_h2d(tensor_t t_host) {
  uint capacity = tensor_get_capacity(t_host);
  T* d_data = (T*)mem_alloc_device(
      capacity * sizeof(T));  // raw data at gpu mem in flat format
  cudaMemcpy(d_data, t_host.data, capacity * sizeof(T), cudaMemcpyHostToDevice);

  tensor_t t_device;
  t_device.data = d_data;
  t_device.dim = t_host.dim;
  t_device.mem_type = GPU_MEM;

  return t_device;
}

void tensor_copy_d2h(tensor_t t_host, tensor_t t_device) {
  assert(t_device.mem_type == GPU_MEM);
  assert(t_host.mem_type == CPU_MEM);

  uint capacity = tensor_get_capacity(t_device);
  AWNN_CHECK_EQ(tensor_get_capacity(t_host), capacity)
  cudaMemcpy(t_host.data, t_device.data, capacity * sizeof(T),
             cudaMemcpyDeviceToHost);
}

void tensor_destroy_device(tensor_t* ptr_t_device) {
  assert(ptr_t_device->mem_type == GPU_MEM);

  mem_free_device(ptr_t_device->data);
}
