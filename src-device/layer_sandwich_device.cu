#include "awnn/common.h"

#include "awnndevice/cublas_wrappers.cuh"
#include "awnndevice/device_utils.cuh"
#include "awnndevice/layer_sandwich_device.cuh"

status_t conv_relu_forward_device(cublasHandle_t handle, tensor_t const d_x,
                                  tensor_t d_w, lcache_t* cache,
                                  conv_param_t const params, tensor_t d_y) {
  return S_OK;
}

status_t conv_relu_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                   tensor_t d_dw, lcache_t* cache,
                                   conv_param_t const params,
                                   tensor_t const d_dy) {
  return S_OK;
}

status_t conv_iden_relu_forward_device(
    cublasHandle_t handle, tensor_t const d_x, tensor_t const d_iden,
    tensor_t d_w, lcache_t* cache, conv_param_t const params, tensor_t d_y) {
  return S_OK;
}

status_t conv_iden_relu_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                        tensor_t d_diden, tensor_t d_dw,
                                        lcache_t* cache,
                                        conv_param_t const params,
                                        tensor_t const d_dy) {
  return S_OK;
}

status_t resblock_forward_device(cublasHandle_t handle, tensor_t const d_x,
                                 tensor_t d_w1, tensor_t d_w2, lcache_t* cache,
                                 conv_param_t const params, tensor_t d_y) {
  tensor_t d_tmp = tensor_make_alike_device(d_y);
  conv_relu_forward_device(handle, d_x, d_w1, cache, params, d_tmp);
  conv_iden_relu_forward_device(handle, d_tmp, d_x, d_w2, cache, params, d_y);

  tensor_destroy_device(&d_tmp);
  return S_OK;
}
status_t resblock_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                  tensor_t d_dw1, tensor_t d_dw2,
                                  lcache_t* cache, conv_param_t const params,
                                  tensor_t const d_dy) {
  tensor_t d_tmp = tensor_make_alike_device(d_dy);
  tensor_t d_dx_iden = tensor_make_alike_device(d_dx);

  conv_iden_relu_backward_device(handle, d_tmp, d_dx_iden, d_dw2, cache, params,
                                 d_dy);
  conv_relu_backward_device(handle, d_dx, d_dw1, cache, params, d_tmp);

  // tensor_elemwise_op_inplace_device(d_dx, d_dx_iden, TENSOR_OP_ADD);
  tensor_destroy(&d_tmp);
  tensor_destroy(&d_dx_iden);

  return S_OK;
}
