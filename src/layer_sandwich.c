#include "awnn/layer_fc.h"
#include "awnn/layer_relu.h"
#include "awnn/layer_sandwich.h"

status_t layer_fc_relu_forward(tensor_t const x, tensor_t const w, tensor_t b,
                               lcache_t *cache, tensor_t y) {
  status_t ret = S_ERR;
  tensor_t tmp = tensor_make_alike(y);

  AWNN_CHECK_EQ(S_OK, layer_fc_forward(x, w, b, cache, tmp));
  AWNN_CHECK_EQ(S_OK, layer_relu_forward(tmp, cache, y));

  ret = S_OK;
  tensor_destroy(tmp);
  return ret;
}

status_t layer_fc_relu_backward(tensor_t dx, tensor_t dw, tensor_t db,
                                lcache_t *cache, tensor_t const dy) {
  status_t ret = S_ERR;

  tensor_t tmp = tensor_make_alike(dy);

  AWNN_CHECK_EQ(S_OK, layer_relu_backward(tmp, cache, dy));
  AWNN_CHECK_EQ(S_OK, layer_fc_backward(dx, dw, db, cache, tmp));

  tensor_destroy(tmp);
  ret = S_OK;
  return ret;
}
