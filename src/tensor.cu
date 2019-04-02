#include "awnn/tensor.h"

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

tensor_t tensor_make_copy_h2d(tensor_t t_host) {
  uint capacity = tensor_get_capacity(t_host);
  T* d_data = (T*)mem_alloc_device(
      capacity * sizeof(T));  // raw data at gpu mem in flat format
  cudaMemcpy(d_data, t_host.data, capacity * sizeof(T), cudaMemcpyHostToDevice);

  tensor_t t_device;
  t_device.data = d_data;
  t_device.dim = t_host.dim;

  return t_device;
}

tensor_t tensor_destroy_device(tensor_t t_device) {
  if (t_device.data) {
    mem_free_device(t_device.data);
  }
}
