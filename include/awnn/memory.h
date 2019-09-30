/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/common.h"
#include "awnn/logging.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
int INC_TOTAL_TENSOR_ALLOC_HOST();
int INC_TOTAL_TENSOR_DEALLOC_HOST();

int GET_TOTAL_TENSOR_ALLOC_HOST();
int GET_TOTAL_TENSOR_DEALLOC_HOST();

void print_memory_alloc_dealloc_totals_host();
int reset_TOTAL_TENSOR_ALLOC_HOST();
int reset_TOTAL_TENSOR_DEALLOC_HOST();
void reset_all_tensor_device_alloc_dealloc_stats_host();
#endif


static inline void* mem_alloc(size_t size) {
  void* ret = malloc(size);
  if (!ret) {
    PERR("Allocation failed\n");
    print_trace();
  } else {
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
    INC_TOTAL_TENSOR_ALLOC_HOST();
#endif
  }
  return ret;
}
static inline void mem_free(void* data) {
  if (data) {
    free(data);
#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
    INC_TOTAL_TENSOR_DEALLOC_HOST();
#endif
  }
}

#ifdef __cplusplus
}
#endif
