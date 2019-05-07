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
  int i;
  for (i = 0; i < cache->count; i++) {
    tensor_destroy(&cache->all_tensors[i]);
  }
  cache->count = 0;
}

void lcache_dump_stat(lcache_t *cache) {
  int i;
  PINF("\n-----Cache Stat ----");
  PINF("--- count= %u -", cache->count);
  for (i = 0; i < cache->count; i++) {
    dim_t dim = cache->all_tensors[i].dim;
    void *addr = cache->all_tensors[i].data;
    int mem_type = (int)cache->all_tensors[i].mem_type;
    PINF("--- [%u]: dim (%u,%u,%u, %u), addr %p, mem_type %d", i, dim.dims[0],
         dim.dims[1], dim.dims[2], dim.dims[3], addr ,mem_type);
  }
  PINF("------   END     ----");
}

// Update the gradient of a parameter if regulizer term exists
void update_regulizer_gradient(tensor_t x, tensor_t dx, T reg) {
  tensor_t tmp = tensor_make_copy(x);
  T *pelem;
  int ii;  // for iteration
  tensor_for_each_entry(pelem, ii, tmp) { (*pelem) *= reg; }
  tensor_elemwise_op_inplace(dx, tmp, TENSOR_OP_ADD);
  tensor_destroy(&tmp);
}
