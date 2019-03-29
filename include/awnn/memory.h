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

void* mem_alloc(size_t size){
  void * ret = malloc(size);
  if(!ret){
    PERR("Allocation faild\n");
  }
}
void mem_free(void* data){
  if(data) free(data);
}

#ifdef __cplusplus
}
#endif
