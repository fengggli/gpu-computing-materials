//
// Created by cmgoebel on 5/5/19.
//

#include "awnn/tensor.h"

#include "awnndevice/memory.cuh"
#include "awnndevice/tensor.cuh"
#include "awnndevice/device_utils.cuh"

#include <cuda_runtime_api.h>  // cudaMemset


tensor_t tensor_make_empty_device(dim_t dim) {
  tensor_t t;
  t.mem_type = GPU_MEM;
  t.dim = dim;
  t.data = NULL;
  t.allocation_tag = -1;

  return t;
}

tensor_t _tensor_make_device(dim_t dim) {
  tensor_t t = tensor_make_empty_device(dim);
  int ret = mem_alloc_device(&t);

  AWNN_CHECK_NE(NULL, t.data);

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
  tensor_t d_t = tensor_make_empty_device(t_host.dim);
  int ret = mem_alloc_device(&d_t);

  cudaMemcpy(d_t.data, t_host.data, tensor_get_capacity(d_t) * sizeof(T), cudaMemcpyHostToDevice);

  return d_t;
}

void tensor_copy_d2h(tensor_t t_host, tensor_t t_device) {
  assert(t_device.mem_type == GPU_MEM);
  assert(t_host.mem_type == CPU_MEM);

  int capacity = tensor_get_capacity(t_device);
  AWNN_CHECK_EQ(tensor_get_capacity(t_host), capacity)
  cudaMemcpy(t_host.data, t_device.data, capacity * sizeof(T),
             cudaMemcpyDeviceToHost);
}

void tensor_destroy_device(tensor_t *t) {
  assert(t->mem_type == GPU_MEM);
  assert(t->data);

  int ret = mem_free_device(t);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
  if(ret != S_OK) {
    PERR("FAILED to destroy device tensor tag %d\n", t->allocation_tag);
  }
#endif
}
