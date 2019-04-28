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

// Update the gradient of a parameter if regulizer term exists
void update_regulizer_gradient(tensor_t x, tensor_t dx, T reg) {
  tensor_t tmp = tensor_make_copy(x);
  T *pelem;
  uint ii;  // for iteration
  tensor_for_each_entry(pelem, ii, tmp) { (*pelem) *= reg; }
  tensor_elemwise_op_inplace(dx, tmp, TENSOR_OP_ADD);
  tensor_destroy(&tmp);
}
