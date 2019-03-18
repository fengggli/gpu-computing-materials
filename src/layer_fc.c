#include "awnn/layer_fc.h"

/* y = x*W+b */
/* https://github.com/fengggli/cs231n-assignments/blob/d4cbe582a794a5b33d81a1ecdb64f1fd3844eaaa/assignment2/FullyConnectedNets.ipynb */
status_t layer_fc_forward(tensor_t const x, tensor_t const w, tensor_t const b, lcache_t *cache, tensor_t y){
  dim_t x_dim;
  tensor_t x_reshaped = tensor_make_copy(x);
  uint N = x.dim.dims[0];
  uint const flat_shape[] = {N, tensor_get_capacity(x)/N};
  tensor_reshape_(&x_reshaped, flat_shape, 2);

  tensor_matmul(x_reshaped, w, y); // y = x*w
  tensor_add_vector_inplace(y, b);

  // create cache for backprog
  tensor_t cached_x = tensor_make_copy(x);
  cache->all_tensors[0] = cached_x;
  cache->count +=1;
  tensor_t cached_w = tensor_make_copy(w);
  cache->all_tensors[1] = cached_w;
  cache->count +=1;
  return S_OK;

}

status_t layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t db, lcache_t *cache, tensor_t const dy) {

  free_lcache(cache);

}

