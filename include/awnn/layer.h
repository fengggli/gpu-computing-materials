/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "awnn/tensor.h"

/* layer cache 
 *
 * Currently each forward will allocate its own cache, and used by its backward
 * TODO: this can be dynamically expanded.
 * */
typedef struct{
  uint count; // number of tensors
  tensor_t * all_tensors;
}lcache_t;

lcache_t make_lcache(){
  lcache_t cache;
  cache.count = 0;
  cache.all_tensors = 0;
  return cache;
}

#endif
