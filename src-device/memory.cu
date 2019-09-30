//
// Created by cmgoebel on 5/6/19.
//

#include "awnndevice/memory.cuh"

#include <cuda_runtime_api.h>  // cudatError_t, cudaMalloc, cudaSuccess, cudaFree

static int TOTAL_TENSOR_ALLOC_DEVICE    = 0;
static int TOTAL_TENSOR_DEALLOC_DEVICE  = 0;

int INC_TOTAL_TENSOR_ALLOC_DEVICE() {
  return ++TOTAL_TENSOR_ALLOC_DEVICE;
}
int INC_TOTAL_TENSOR_DEALLOC_DEVICE() {
  return ++TOTAL_TENSOR_DEALLOC_DEVICE;
}

int GET_TOTAL_TENSOR_ALLOC_DEVICE() {
  return TOTAL_TENSOR_ALLOC_DEVICE;
}
int GET_TOTAL_TENSOR_DEALLOC_DEVICE() {
  return TOTAL_TENSOR_DEALLOC_DEVICE;
}

void print_memory_alloc_dealloc_totals_device() {
  printf("total device allocations = %d, total device de-allocations = %d\n",
         TOTAL_TENSOR_ALLOC_DEVICE, TOTAL_TENSOR_DEALLOC_DEVICE);
}

int reset_TOTAL_TENSOR_ALLOC_DEVICE() {
  TOTAL_TENSOR_ALLOC_DEVICE = 0;
  return TOTAL_TENSOR_ALLOC_DEVICE;
}
int reset_TOTAL_TENSOR_DEALLOC_DEVICE() {
  TOTAL_TENSOR_DEALLOC_DEVICE = 0;
  return TOTAL_TENSOR_DEALLOC_DEVICE;
}
void reset_all_tensor_device_alloc_dealloc_stats_device() {
  TOTAL_TENSOR_ALLOC_DEVICE = 0;
  TOTAL_TENSOR_DEALLOC_DEVICE = 0;
}


int mem_alloc_device(tensor_t *d_t) {
  assert(d_t->mem_type == GPU_MEM);

  int size = tensor_get_capacity(*d_t) * (int)sizeof(T);

  cudaError_t cudaStat;
  cudaStat = cudaMalloc((void**)&d_t->data, size);
  AWNN_CHECK_EQ(cudaStat, cudaSuccess);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
  if(cudaStat == cudaSuccess) {
    d_t->allocation_tag = GET_TOTAL_TENSOR_ALLOC_DEVICE();
    INC_TOTAL_TENSOR_ALLOC_DEVICE();
    printf("allocated device tensor %d\n", d_t->allocation_tag);
  }
#endif
  return S_OK;
}

int mem_free_device(tensor_t *d_t) {
  if (d_t->data) {
    cudaError_t stat = cudaFree(d_t->data);
//    AWNN_CHECK_EQ(stat, cudaSuccess);
    d_t->data = NULL;
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
    INC_TOTAL_TENSOR_DEALLOC_DEVICE();
    if (stat != cudaSuccess) {
      printf("deallocated device tensor FAILURE %d\n", d_t->allocation_tag);
    } else {
      printf("deallocated device tensor %d\n", d_t->allocation_tag);

    }
#endif
    return S_OK;
  }
  return S_ERR;
}