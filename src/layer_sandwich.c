#include "awnn/layer_sandwich.h"

status_t fc_relu_forward(tensor_t const x, tensor_t const w, tensor_t b,
                         lcache_t *cache, tensor_t y) {
  status_t ret = S_ERR;
  tensor_t tmp = tensor_make_alike(y);

  AWNN_CHECK_EQ(S_OK, layer_fc_forward(x, w, b, cache, tmp));
  AWNN_CHECK_EQ(S_OK, layer_relu_forward(tmp, cache, y));

  ret = S_OK;
  tensor_destroy(&tmp);
  return ret;
}

status_t fc_relu_backward(tensor_t dx, tensor_t dw, tensor_t db,
                          lcache_t *cache, tensor_t const dy) {
  status_t ret = S_ERR;

  tensor_t tmp = tensor_make_alike(dy);

  AWNN_CHECK_EQ(S_OK, layer_relu_backward(tmp, cache, dy));
  AWNN_CHECK_EQ(S_OK, layer_fc_backward(dx, dw, db, cache, tmp));

  tensor_destroy(&tmp);
  ret = S_OK;
  return ret;
}

status_t conv_relu_forward(tensor_t const x, tensor_t const w, lcache_t *cache,
                           conv_param_t const params, tensor_t y) {
  status_t ret = S_ERR;
  tensor_t tmp = tensor_make_alike(y);

  AWNN_CHECK_EQ(S_OK, convolution_forward(x, w, cache, params, tmp));
  AWNN_CHECK_EQ(S_OK, layer_relu_forward(tmp, cache, y));

  ret = S_OK;
  tensor_destroy(&tmp);
  return ret;
}

status_t conv_relu_backward(tensor_t dx, tensor_t dw, lcache_t *cache,
                            conv_param_t const params, tensor_t const dy) {
  status_t ret = S_ERR;

  tensor_t tmp = tensor_make_alike(dy);

  AWNN_CHECK_EQ(S_OK, layer_relu_backward(tmp, cache, dy));
  AWNN_CHECK_EQ(S_OK, convolution_backward(dx, dw, cache, params, tmp));

  tensor_destroy(&tmp);
  ret = S_OK;
  return ret;
}

status_t conv_iden_relu_forward(tensor_t x, tensor_t iden, tensor_t w,
                                lcache_t *cache, conv_param_t params,
                                tensor_t y) {
  AWNN_CHECK_EQ(
      x.dim.dims[3],
      y.dim.dims[3]);  // in resnet tensor h/w doesn't change in each stage
  tensor_t tmp = tensor_make_alike(y);
  AWNN_CHECK_EQ(S_OK, convolution_forward(x, w, cache, params, tmp));
  tensor_elemwise_op_inplace(tmp, iden, TENSOR_OP_ADD);
  AWNN_CHECK_EQ(S_OK, layer_relu_forward(tmp, cache, y));
  tensor_destroy(&tmp);
  return S_OK;
}

status_t conv_iden_relu_backward(tensor_t dx, tensor_t diden, tensor_t dw,
                                 lcache_t *cache, conv_param_t params,
                                 tensor_t dy) {
  // tensor_t tmp = tensor_make_alike(dy);

  AWNN_CHECK_EQ(S_OK, layer_relu_backward(diden, cache, dy));
  // tensor_copy(diden, tmp);
  AWNN_CHECK_EQ(S_OK, convolution_backward(dx, dw, cache, params, diden));

  // tensor_destroy(&tmp);
  return S_OK;
}

status_t residual_basic_no_bn_forward(tensor_t x, tensor_t w1, tensor_t w2,
                                      lcache_t *cache,
                                      conv_param_t const params, tensor_t y) {
  AWNN_CHECK_EQ(
      x.dim.dims[3],
      y.dim.dims[3]);  // in resnet tensor h/w doesn't change in each stage
  tensor_t tmp = tensor_make_alike(y);
  conv_relu_forward(x, w1, cache, params, tmp);
  conv_iden_relu_forward(tmp, x, w2, cache, params, y);
  tensor_destroy(&tmp);

  return S_OK;
}

status_t residual_basic_no_bn_backward(tensor_t dx, tensor_t dw1, tensor_t dw2,
                                       lcache_t *cache,
                                       conv_param_t const params, tensor_t dy) {
  tensor_t tmp = tensor_make_alike(dy);
  tensor_t dx_iden = tensor_make_alike(dx);

  conv_iden_relu_backward(tmp, dx_iden, dw2, cache, params, dy);
  conv_relu_backward(dx, dw1, cache, params, tmp);

  tensor_elemwise_op_inplace(dx, dx_iden, TENSOR_OP_ADD);
  tensor_destroy(&tmp);
  tensor_destroy(&dx_iden);

  return S_OK;
}
