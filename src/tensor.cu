#include "awnn/tensor.h"
#include "cuda_defs.h"

void* mem_alloc_device(size_t size) {
  T* d_data;
  cudaError_t cudaStat;
  cudaStat = cudaMalloc(&d_data, size);
  AWNN_CHECK_EQ(cudaStat, cudaSuccess);
  return d_data;
}

void mem_free_device(void* d_data) {
  if (d_data) cudaFree(d_data);
}

tensor_t tensor_make_device(uint const shape[], uint const ndims) {
  tensor_t t_device;

  uint capacity = 1;
  for (int i = 0; i < ndims; ++i) {
    capacity *= shape[i];
    t_device.dim.dims[i] = shape[i];
  }

  T* d_data = (T*)mem_alloc_device(capacity * sizeof(T));  // raw data at gpu mem in flat format


  t_device.data = d_data;
  t_device.mem_type = GPU_MEM;

  return t_device;
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
  uint capacity = tensor_get_capacity(t_device);
  AWNN_CHECK_EQ(tensor_get_capacity(t_host), capacity)
  cudaMemcpy(t_host.data, t_device.data, capacity * sizeof(T),
             cudaMemcpyDeviceToHost);
}

void tensor_destroy_device(tensor_t* ptr_t_device) {
  mem_free_device(ptr_t_device->data);
}
