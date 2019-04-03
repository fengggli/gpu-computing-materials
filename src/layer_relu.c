#include "awnn/layer_relu.h"
#include "awnn/logging.h"

status_t layer_relu_forward(tensor_t const x, lcache_t *cache, tensor_t y) {
  status_t ret = S_ERR;

  tensor_t mask = tensor_make_alike(x);
  for (uint i = 0; i < tensor_get_capacity(mask); i++) {
    mask.data[i] = x.data[i] > 0 ? 1.0 : 0.0;
  }

  tensor_copy(y, x);
  tensor_elemwise_op_inplace(y, mask, TENSOR_OP_MUL);

  if (cache) {
    lcache_push(cache, mask);
  }

  ret = S_OK;
  return ret;
}

status_t layer_relu_backward(tensor_t dx, lcache_t *cache, tensor_t const dy) {
  status_t ret = S_ERR;

  tensor_t mask = lcache_pop(cache);

  tensor_copy(dx, dy);
  tensor_elemwise_op_inplace(dx, mask, TENSOR_OP_MUL);

  ret = S_OK;

  tensor_destroy(&mask);
  return ret;
}
