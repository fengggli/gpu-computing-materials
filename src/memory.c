//
// Created by cmgoebel on 4/29/19.
//

#include "memory.h"

#include <stdio.h>

static int TOTAL_TENSOR_ALLOC_HOST = 0;
static int TOTAL_TENSOR_DEALLOC_HOST = 0;

int INC_TOTAL_TENSOR_ALLOC_HOST() {
  ++TOTAL_TENSOR_ALLOC_HOST;
  return TOTAL_TENSOR_ALLOC_HOST;
}
int INC_TOTAL_TENSOR_DEALLOC_HOST() {
  ++TOTAL_TENSOR_DEALLOC_HOST;
  return TOTAL_TENSOR_DEALLOC_HOST;
}

int GET_TOTAL_TENSOR_ALLOC_HOST() {
  return TOTAL_TENSOR_ALLOC_HOST;
}
int GET_TOTAL_TENSOR_DEALLOC_HOST() {
  return TOTAL_TENSOR_DEALLOC_HOST;
}

int reset_TOTAL_TENSOR_ALLOC_HOST() {
  TOTAL_TENSOR_ALLOC_HOST = 0;
  return TOTAL_TENSOR_ALLOC_HOST;
}
int reset_TOTAL_TENSOR_DEALLOC_HOST() {
  TOTAL_TENSOR_DEALLOC_HOST = 0;
  return TOTAL_TENSOR_DEALLOC_HOST;
}

void print_memory_alloc_dealloc_totals_host() {
  printf("total host allocations = %d, total host de-allocations = %d\n",
         TOTAL_TENSOR_ALLOC_HOST, TOTAL_TENSOR_DEALLOC_HOST);
}

void reset_all_tensor_device_alloc_dealloc_stats_host() {
  TOTAL_TENSOR_ALLOC_HOST = 0;
  TOTAL_TENSOR_DEALLOC_HOST = 0;
}