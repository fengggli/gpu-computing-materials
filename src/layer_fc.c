#include "awnn/layer_fc.h"
#include "awnn/logging.h"

#ifdef USE_OPENBLAS
#include "cblas.h"
#endif
#ifdef USE_MKL
#include "mkl.h"
#endif


/* y = x*W+b */
/* https://github.com/fengggli/cs231n-assignments/blob/d4cbe582a794a5b33d81a1ecdb64f1fd3844eaaa/assignment2/FullyConnectedNets.ipynb
 */
void do_layer_fc_forward(tensor_t const x, tensor_t const w, tensor_t const b,
                         tensor_t y) {
  // flatten x to from N-d to 2-d
  uint N = x.dim.dims[0];
  uint const flat_shape[] = {N, tensor_get_capacity(x) / N};

  // shadow copy
  tensor_t x_reshaped = tensor_make_placeholder(flat_shape, 2);
  x_reshaped.data = x.data;

  // forwarding
  tensor_matmul(x_reshaped, w, y);  // y = x*w
  tensor_add_vector_inplace(y, b);
}

status_t layer_fc_forward(tensor_t const x, tensor_t const w, tensor_t const b,
                          lcache_t *cache, tensor_t y) {
  do_layer_fc_forward(x, w, b, y);
  if (cache) {
    lcache_push(cache, x);
    lcache_push(cache, w);
  }
  return S_OK;
}

void do_layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t db,
                          tensor_t const dy, tensor_t x, tensor_t w) {
  // y = x*w+b  dy/dx = w
  // dy ~ dL/dy
  // * dL/dx
  // * dL/dw

  // calculate gradient: dx = dy*w^T
  uint N = dx.dim.dims[0];
  uint const flat_shape[] = {N, tensor_get_capacity(dx) / N};
  dim_t old_dim = dx.dim;  // save to recover later
  AWNN_CHECK_EQ(S_OK, tensor_reshape_(&dx, flat_shape, 2));
  tensor_matmul_full(dy, CblasNoTrans, w, CblasTrans, dx);
  dx.dim = old_dim;  // reshape to N-d!

  // shadow copy
  tensor_t x_reshaped = tensor_make_placeholder(flat_shape, 2);
  x_reshaped.data = x.data;

  // calculate gradient dw = x^T * dy
  tensor_matmul_full(x_reshaped, CblasTrans, dy, CblasNoTrans, dw);

  // gradient db = sum(dy, axis =1)
  // this make [N, M] sum to [1,M];
  uint axis_id = 0;
  tensor_t sum = tensor_make_sum(dy, axis_id);
  uint const tmp_shape[] = {tensor_get_capacity(db)};
  AWNN_CHECK_EQ(S_OK, tensor_reshape_(&sum, tmp_shape, 1));  // reshape to 2d
  tensor_copy(db, sum);
  tensor_destroy(&sum);
}

status_t layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t db,
                           lcache_t *cache, tensor_t const dy) {
  tensor_t x, w;
  w = lcache_pop(cache);
  x = lcache_pop(cache);
  do_layer_fc_backward(dx, dw, db, dy, x, w);
  return S_OK;
}
