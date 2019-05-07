#include "awnn/tensor.h"
#include "awnndevice/device_utils.cuh"

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
#include "awnndevice/memory.cuh"
#endif


#include <cuda_runtime_api.h>  // cudaMemset

void* mem_alloc_device(size_t size) {
  T* d_data;
  cudaError_t cudaStat;
  cudaStat = cudaMalloc(&d_data, size);
  AWNN_CHECK_EQ(cudaStat, cudaSuccess);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
  if(cudaStat == cudaSuccess) {
    INC_TOTAL_TENSOR_ALLOC_DEVICE();
  }
#endif
  return d_data;
}

int mem_free_device(void* d_data) {
  if (d_data) {
    cudaFree(d_data);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
    INC_TOTAL_TENSOR_DEALLOC_DEVICE();
#endif
    return S_OK;
  }
  return S_ERR;
}

tensor_t _tensor_make_device(dim_t dim) {
  tensor_t t;
  int capacity;
  capacity = dim_get_capacity(dim);
  t.data = (T *)mem_alloc_device(capacity * sizeof(T));
  t.mem_type = GPU_MEM;
  t.dim = dim;
  AWNN_CHECK_NE(NULL, t.data);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
  int a = GET_TOTAL_TENSOR_ALLOC_DEVICE() - 1;
  t.allocation_tag = a;
  printf("allocated d tensor with tag %d\n", t.allocation_tag);
  if(a)
    printf ("%d\n", a);
  else{
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>a is 0");
  }
#endif
  return t;
}

tensor_t tensor_make_alike_device(tensor_t t) {
  AWNN_CHECK_EQ(t.mem_type, GPU_MEM);
  return _tensor_make_device(t.dim); 
}

tensor_t tensor_make_device(int const shape[], int const ndims) {
  int i;
  dim_t dim;

  if (ndims == 0) {
    PINF("make zero");
    dim = make_dim(0, 0);
  }

  for (i = 0; i < MAX_DIM; i++) {
    if (i < ndims)
      dim.dims[i] = shape[i];
    else
      dim.dims[i] = 0;
  }
  return _tensor_make_device(dim);
}

tensor_t tensor_make_zeros_device(int const shape[], int const ndims) {
  tensor_t t = tensor_make_device(shape, ndims);
  cudaMemset(t.data, 0, tensor_get_capacity(t) * sizeof(T));

  return t;
}

tensor_t tensor_make_copy_h2d(tensor_t t_host) {
  assert(t_host.mem_type == CPU_MEM);
  int capacity = tensor_get_capacity(t_host);
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

  int capacity = tensor_get_capacity(t_device);
  AWNN_CHECK_EQ(tensor_get_capacity(t_host), capacity)
  cudaMemcpy(t_host.data, t_device.data, capacity * sizeof(T),
             cudaMemcpyDeviceToHost);
}

void tensor_destroy_device(tensor_t* ptr_t_device) {
  assert(ptr_t_device->mem_type == GPU_MEM);

  int ret = mem_free_device(ptr_t_device->data);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
  if (ret == S_OK) {
    printf("deallocate d tensor with tag = %d\n", ptr_t_device->allocation_tag);
  }
#endif

}
