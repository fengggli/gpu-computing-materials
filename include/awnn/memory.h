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


long INC_TOTAL_TENSOR_ALLOC_HOST();
long INC_TOTAL_TENSOR_ALLOC_DEVICE();
long INC_TOTAL_TENSOR_DEALLOC_HOST();
long INC_TOTAL_TENSOR_DEALLOC_DEVICE();

long GET_TOTAL_TENSOR_ALLOC_HOST();
long GET_TOTAL_TENSOR_DEALLOC_HOST();
long GET_TOTAL_TENSOR_ALLOC_DEVICE();
long GET_TOTAL_TENSOR_DEALLOC_DEVICE();



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

void* mem_alloc_device(size_t size);

void mem_free_device(void* data);



#ifdef __cplusplus
}
#endif
