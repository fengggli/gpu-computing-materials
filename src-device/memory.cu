//
// Created by cmgoebel on 5/6/19.
//

#include "awnndevice/memory.cuh"

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

void print_memory_alloc_dealloc_totals() {
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
void reset_all_tensor_device_alloc_dealloc_stats() {
  TOTAL_TENSOR_ALLOC_DEVICE = 0;
  TOTAL_TENSOR_DEALLOC_DEVICE = 0;
}