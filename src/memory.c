//
// Created by cmgoebel on 4/29/19.
//
#include "memory.h"


static long TOTAL_TENSOR_ALLOC_HOST = 0;
static long TOTAL_TENSOR_ALLOC_DEVICE = 0;
static long TOTAL_TENSOR_DEALLOC_HOST = 0;
static long TOTAL_TENSOR_DEALLOC_DEVICE = 0;

long INC_TOTAL_TENSOR_ALLOC_HOST() {
//  printf("host allocate called\n");
  return ++TOTAL_TENSOR_ALLOC_HOST;
}
long INC_TOTAL_TENSOR_ALLOC_DEVICE() {
//  printf("device allocate called\n");
  return ++TOTAL_TENSOR_ALLOC_DEVICE;
}
long INC_TOTAL_TENSOR_DEALLOC_HOST() {
//  printf("host de-allocate called\n");
  return ++TOTAL_TENSOR_DEALLOC_HOST;
}
long INC_TOTAL_TENSOR_DEALLOC_DEVICE() {
//  printf("device de-allocate called\n");
  return ++TOTAL_TENSOR_DEALLOC_DEVICE;
}


long GET_TOTAL_TENSOR_ALLOC_HOST() {
  return TOTAL_TENSOR_ALLOC_HOST;
}
long GET_TOTAL_TENSOR_DEALLOC_HOST() {
  return TOTAL_TENSOR_DEALLOC_HOST;
}
long GET_TOTAL_TENSOR_ALLOC_DEVICE() {
  return TOTAL_TENSOR_ALLOC_DEVICE;
}
long GET_TOTAL_TENSOR_DEALLOC_DEVICE() {
  return TOTAL_TENSOR_DEALLOC_DEVICE;
}

void print_memory_alloc_dealloc_totals() {
  printf("total host allocations = %ld, total host de-allocations = %ld\n"
         "total device allocations = %ld, total_device de-allocations = %ld\n",
         TOTAL_TENSOR_ALLOC_HOST, TOTAL_TENSOR_DEALLOC_HOST,
         TOTAL_TENSOR_ALLOC_DEVICE, TOTAL_TENSOR_DEALLOC_DEVICE);
}

long reset_TOTAL_TENSOR_ALLOC_HOST() {
  TOTAL_TENSOR_ALLOC_HOST = 0;
  return TOTAL_TENSOR_ALLOC_HOST;
}
long reset_TOTAL_TENSOR_DEALLOC_HOST() {
  TOTAL_TENSOR_DEALLOC_HOST = 0;
  return TOTAL_TENSOR_DEALLOC_HOST;
}
long reset_TOTAL_TENSOR_ALLOC_DEVICE() {
  TOTAL_TENSOR_ALLOC_DEVICE = 0;
  return TOTAL_TENSOR_ALLOC_DEVICE;
}
long reset_TOTAL_TENSOR_DEALLOC_DEVICE() {
  TOTAL_TENSOR_DEALLOC_DEVICE = 0;
  return TOTAL_TENSOR_DEALLOC_DEVICE;
}
void reset_all_tensor_alloc_dealloc_stats() {
  TOTAL_TENSOR_ALLOC_HOST = 0;
  TOTAL_TENSOR_DEALLOC_HOST = 0;
  TOTAL_TENSOR_ALLOC_DEVICE = 0;
  TOTAL_TENSOR_DEALLOC_DEVICE = 0;
}