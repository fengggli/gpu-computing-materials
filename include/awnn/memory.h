/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include <stdlib.h>
#include "awnn/common.h"
#include "awnn/logging.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void* mem_alloc(size_t size) {
  void* ret = malloc(size);
  if (!ret) {
    PERR("Allocation faild\n");
    print_trace();
  }
  return ret;
}
static inline void mem_free(void* data) {
  if (data) free(data);
}

void* mem_alloc_device(size_t size);

void mem_free_device(void* data);

#ifdef __cplusplus
}
#endif
