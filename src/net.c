#include "awnn/layer.h"
#include "awnn/net.h"
void lcache_push(lcache_t *cache, tensor_t t) {
  cache->all_tensors[cache->count++] = t;
}

tensor_t lcache_pop(lcache_t *cache) {
  tensor_t ret = cache->all_tensors[cache->count - 1];
  cache->count--;
  return ret;
}

void make_empty_lcache(lcache_t *cache) { cache->count = 0; }

/* This is merely needed most of the time, we should pop and destroy one by
 * one*/
void lcache_free_all(lcache_t *cache) {
  uint i;
  for (i = 0; i < cache->count; i++) {
    tensor_destroy(&cache->all_tensors[i]);
  }
  cache->count = 0;
}
