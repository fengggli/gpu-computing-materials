#include "awnn/layer_fc.h"

/* y = x*W+b */
status_t layer_fc_forward(tensor_t const x, tensor_t const w, tensor_t const b, lcache_t *cache, tensor_t y){
  dim_t x_dim;
  tensor_t x_reshaped = tensor_make_copy(x);
  uint N = x.dim.dims[0];
  uint const flat_shape[] = {N, tensor_get_capacity(x)/N};
  tensor_reshape_(&x_reshaped, flat_shape, 2);

  tensor_matmul(x_reshaped, w, y); // y = x*w
  // shape_b = tensor_reshape()
  // b_extend = tensor_make_scalar()
  // awnn_plus(1.0, y, b_extend); // y = y + b
}

status_t layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t db, lcache_t *cache, tensor_t const dy) {
}

