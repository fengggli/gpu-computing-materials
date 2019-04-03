#include "awnn/layer_fc.h"
#include "awnn/logging.h"

/* y = x*W+b */
/* https://github.com/fengggli/cs231n-assignments/blob/d4cbe582a794a5b33d81a1ecdb64f1fd3844eaaa/assignment2/FullyConnectedNets.ipynb
 */
status_t layer_fc_forward(tensor_t const x, tensor_t const w, tensor_t const b,
                          lcache_t *cache, tensor_t y) {
  // flatten x to from N-d to 2-d
  tensor_t x_reshaped = tensor_make_copy(x);
  uint N = x.dim.dims[0];
  uint const flat_shape[] = {N, tensor_get_capacity(x) / N};
  tensor_reshape_(&x_reshaped, flat_shape, 2);

  // forwarding
  tensor_matmul(x_reshaped, w, y);  // y = x*w
  tensor_add_vector_inplace(y, b);

  // create cache for backprog
  if (cache) {
    tensor_t cached_x_T =
        tensor_make_transpose(x_reshaped);  // saves transpose of flattened x
    lcache_push(cache, cached_x_T);
    tensor_t cached_w_T = tensor_make_transpose(w);  // saves transpose of W
    lcache_push(cache, cached_w_T);
  }

  // free temprary tensors
  tensor_destroy(&x_reshaped);
  return S_OK;
}

status_t layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t db,
                           lcache_t *cache, tensor_t const dy) {
  status_t ret = S_ERR;

  // y = x*w+b  dy/dx = w
  // dy ~ dL/dy
  // * dL/dx
  // * dL/dw

  // get the cache contents
  tensor_t w_T = lcache_pop(cache);
  tensor_t x_reshaped_T = lcache_pop(cache);

  // calculate gradient: dx = dy*w^T
  uint N = dx.dim.dims[0];
  uint const flat_shape[] = {N, tensor_get_capacity(dx) / N};
  dim_t old_dim = dx.dim;  // save to recover later
  if (S_OK != tensor_reshape_(&dx, flat_shape, 2)) goto end;
  tensor_matmul(dy, w_T, dx);
  dx.dim = old_dim;  // reshape to N-d!

  // calculate gradient dw = x^T * dy
  tensor_matmul(x_reshaped_T, dy, dw);

  // gradient db = sum(dy, axis =1)
  // this make [N, M] sum to [1,M];
  uint axis_id = 0;
  tensor_t sum = tensor_make_sum(dy, axis_id);
  uint const tmp_shape[] = {tensor_get_capacity(db)};
  if (S_OK != tensor_reshape_(&sum, tmp_shape, 1)) goto end;  // reshape to 2d
  tensor_copy(db, sum);
  tensor_destroy(&sum);
  ret = S_OK;

end:
  // free layer cache
  tensor_destroy(&w_T);
  tensor_destroy(&x_reshaped_T);
  return ret;
}
