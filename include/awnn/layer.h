/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "awnn/tensor.h"

#define MAX_CACHE_SIZE (10)
/*
 * @brief layer cache
 *
 * Currently each forward will allocate its own cache, and used by its backward.
 * lcache_t will be created outside of the forward function(using make_lcache);
 * but the count and all_tensors will be populated inside the forward.
 * The reason is that different layers require different number of caches
 *
 * Backward pass needs to delete the cache
 *
 * */
typedef struct{
  uint count; // number of tensors
  tensor_t all_tensors[MAX_CACHE_SIZE];
}lcache_t;

static void make_empty_lcache(lcache_t *cache){
  cache->count = 0;
}

/* This should be called inside the backprop*/
static void free_lcache(lcache_t *cache){
  uint i;
  for(i =0; i <cache->count; i++){
    tensor_destroy(cache->all_tensors[i]);
  }
  cache->count = 0;
}


#endif
