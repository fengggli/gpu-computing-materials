#include "awnn/layer.h"
#include "awnn/net.h"
#include <immintrin.h>
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

void lcache_dump_stat(lcache_t *cache) {
  uint i;
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
  size_t capacity = tensor_get_capacity(x);
  AWNN_CHECK_EQ(capacity, tensor_get_capacity(dx));

#if 1
  AWNN_CHECK_EQ(capacity%8, 0);
  __m256 _old,_new, _reg, _x, _tmp; 
  _reg = _mm256_set_ps(reg, reg, reg, reg, reg, reg, reg, reg);
  for (size_t i = 0; i < capacity; i += 8) {
    // dx.data[i] += reg * (x.data[i]);
    _old = _mm256_load_ps(dx.data + i);
    _x = _mm256_load_ps(x.data + i);
    _tmp = _mm256_mul_ps(_reg, _x);
    _x = _mm256_add_ps(_old, _tmp);
    _mm256_store_ps(dx.data+i, _x);
  }

#else
  for (size_t i = 0; i < capacity; i++) {
    dx.data[i] += reg * (x.data[i]);
  }
#endif
}
