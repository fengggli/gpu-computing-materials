#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"
#include "awnn/layer.h"   // for lcache_t

#include <cublas_v2.h> // for cublasHandle_t

#ifdef __cplusplus
extern "C" {
#endif

struct layer_context_device{
  tensor_t d_tmp;
  tensor_t d_dtmp;
};

void layer_context_destroy_device(struct layer_context_device *context);
status_t relu_forward_device(tensor_t const d_x,
                                  lcache_t* cache,
                                  tensor_t d_y);
status_t relu_backward_device(tensor_t const d_dx,
                                  lcache_t* cache,
                                  tensor_t d_dy);
status_t conv_relu_forward_device(cublasHandle_t handle, tensor_t const d_x,
                                  tensor_t d_w, lcache_t* cache,
                                  conv_param_t const params, tensor_t d_y, struct layer_context_device * context);
status_t conv_relu_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                   tensor_t d_dw, lcache_t* cache,
                                   conv_param_t const params,
                                   tensor_t const d_dy, layer_context_device * context);




status_t resblock_forward_device(cublasHandle_t handle, tensor_t const d_x, tensor_t d_w1, tensor_t d_w2, lcache_t* cache, conv_param_t const params, tensor_t d_y);
status_t resblock_backward_device(cublasHandle_t handle, tensor_t d_dx, tensor_t d_dw1, tensor_t d_dw2, lcache_t* cache, conv_param_t const params, tensor_t const d_dout);

#ifdef __cplusplus
}
#endif

