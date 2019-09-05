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
  // TODO: this is expensive and buggy
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
  tensor_t tmp = tensor_make_alike(dy);

  AWNN_CHECK_EQ(S_OK, layer_relu_backward(tmp, cache, dy));
  tensor_copy(diden, tmp);
  AWNN_CHECK_EQ(S_OK, convolution_backward(dx, dw, cache, params, tmp));

  tensor_destroy(&tmp);
  return S_OK;
}

/**
 * Residual Block without subsampling in the beginning
 *
 * @param x
 * @param w1
 * @param w2
 * @param cache
 * @param conv_param1 the dimension will be checked to handle the sub-sampling
 * in the first block of the stage 2,3,4
 * @param conv_param2
 * @param y
 * @return
 */
status_t residual_basic_no_bn_forward(tensor_t x, tensor_t w1, tensor_t w2,
                                      lcache_t *cache,
                                      conv_param_t const conv_param,
                                      tensor_t y) {
  AWNN_CHECK_EQ(conv_param.stride, 1);
  AWNN_CHECK_EQ(x.dim.dims[3], y.dim.dims[3]);

  tensor_t tmp = tensor_make_alike(y);
  conv_relu_forward(x, w1, cache, conv_param, tmp);
  conv_iden_relu_forward(tmp, x, w2, cache, conv_param, y);

  // TODO: destroyed here
  tensor_destroy(&tmp);

  return S_OK;
}

status_t residual_basic_no_bn_backward(tensor_t dx, tensor_t dw1, tensor_t dw2,
                                       lcache_t *cache,
                                       conv_param_t const conv_param,
                                       tensor_t dy) {
  tensor_t tmp = tensor_make_alike(dy);
  tensor_t dx_iden = tensor_make_alike(dx);

  conv_iden_relu_backward(tmp, dx_iden, dw2, cache, conv_param, dy);
  conv_relu_backward(dx, dw1, cache, conv_param, tmp);

  tensor_elemwise_op_inplace(dx, dx_iden, TENSOR_OP_ADD);
  tensor_destroy(&tmp);
  tensor_destroy(&dx_iden);

  return S_OK;
}

/**
 * Residual Block WITH subsampling in the beginning
 *
 * @param x
 * @param w_sample weight for the subsampling convolution(1x1)
 * @param w1 first 3x3 convolution
 * @param w2 second 3x3 convolution
 * @param cache
 * @param conv_param1 the dimension will be checked to handle the sub-sampling
 * in the first block of the stage 2,3,4
 * @param conv_param2
 * @param y
 * @return
 *
 */
status_t residual_basic_no_bn_subspl_forward(tensor_t x, tensor_t w_sample,
                                             tensor_t w1, tensor_t w2,
                                             lcache_t *cache,
                                             conv_param_t const conv_param1,
                                             conv_param_t const conv_param2,
                                             tensor_t y) {
  AWNN_CHECK_EQ(conv_param1.stride, 2);
  AWNN_CHECK_EQ(x.dim.dims[3], 2 * y.dim.dims[3]);
  AWNN_CHECK_EQ(w_sample.dim.dims[3], 1);

  tensor_t tmp = tensor_make_alike(y);

  tensor_t x_identity = tensor_make_alike(y);               // identity
  conv_param_t subspl_param = {.stride = 2, .padding = 0};  // 1x1 kernel
  convolution_forward(x, w_sample, cache, subspl_param, x_identity);

  conv_relu_forward(x, w1, cache, conv_param1, tmp);
  conv_iden_relu_forward(tmp, x_identity, w2, cache, conv_param2, y);

  tensor_destroy(&tmp);
  tensor_destroy(&x_identity);

  return S_OK;
}

status_t residual_basic_no_bn_subspl_backward(tensor_t dx, tensor_t dw_sample,
                                              tensor_t dw1, tensor_t dw2,
                                              lcache_t *cache,
                                              conv_param_t const conv_param1,
                                              conv_param_t const conv_param2,
                                              tensor_t dy) {
  tensor_t tmp = tensor_make_alike(dy);
  tensor_t dx_iden = tensor_make_alike(dy);
  tensor_t dx_2 = tensor_make_alike(dx);  // gradient of x from identity

  conv_iden_relu_backward(tmp, dx_iden, dw2, cache, conv_param2, dy);
  conv_relu_backward(dx, dw1, cache, conv_param1, tmp);

  conv_param_t subspl_param = {.stride = 2, .padding = 0};  // 1x1 kernel
  convolution_backward(dx_2, dw_sample, cache, subspl_param, dx_iden);

  tensor_elemwise_op_inplace(dx, dx_2, TENSOR_OP_ADD);
  tensor_destroy(&tmp);
  tensor_destroy(&dx_iden);
  tensor_destroy(&dx_2);

  return S_OK;
}
