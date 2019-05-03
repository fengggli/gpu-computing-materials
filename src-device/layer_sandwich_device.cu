#include <stdlib.h>
#include "awnn/common.h"
#include "awnn/memory.h"

#include "awnndevice/cublas_wrappers.cuh"
#include "awnndevice/device_utils.cuh"
#include "awnndevice/layer_conv_device.cuh"
#include "awnndevice/layer_sandwich_device.cuh"

/** Destroy all cache only for this context*/
void layer_context_destroy_device(struct layer_context_device* context) {
  tensor_destroy_device(&context->d_tmp);
  tensor_destroy_device(&context->d_dtmp);
}

__global__ void do_device_relu_forward(tensor_t d_x, tensor_t d_y) {
  for (uint i : grid_stride_range(0u, d_capacity(d_x))) {
    d_y.data[i] = d_x.data[i] > 0 ? d_x.data[i] : 0.0;
  }
}
__global__ void do_device_relu_backward(tensor_t d_dx, tensor_t d_x,
                                        tensor_t d_dy) {
  for (uint i : grid_stride_range(0u, d_capacity(d_x))) {
    d_dx.data[i] = d_x.data[i] > 0 ? d_dy.data[i] : 0.0;
  }
}

status_t relu_forward_device(tensor_t const d_x, lcache_t* cache,
                             tensor_t d_y) {
  do_device_relu_forward<<<32, 1024>>>(d_x, d_y);

  if (cache) {
    lcache_push(cache, d_x);
  }
  return S_OK;
}
status_t relu_backward_device(tensor_t const d_dx, lcache_t* cache,
                              tensor_t d_dy) {
  // lcache_dump_stat(cache);
  tensor_t d_x = lcache_pop(cache);
  do_device_relu_backward<<<32, 1024>>>(d_dx, d_x, d_dy);

  return S_OK;
}

status_t conv_relu_forward_device(cublasHandle_t handle, tensor_t const d_x,
                                  tensor_t d_w, lcache_t* cache,
                                  conv_param_t const params, tensor_t d_y,
                                  struct layer_context_device* context) {
  AWNN_CHECK_EQ(d_x.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_w.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_y.mem_type, GPU_MEM);
  tensor_t d_tmp = context->d_tmp;

  AWNN_CHECK_EQ(
      S_OK, convolution_forward_device(handle, d_x, d_w, cache, params, d_tmp));
  AWNN_CHECK_EQ(S_OK, relu_forward_device(d_tmp, cache, d_y));

  // lcache_dump_stat(cache);
  return S_OK;
}

status_t conv_relu_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                   tensor_t d_dw, lcache_t* cache,
                                   conv_param_t const params,
                                   tensor_t const d_dy,
                                   struct layer_context_device* context) {
  AWNN_CHECK_EQ(d_dx.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dw.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dy.mem_type, GPU_MEM);

  PINF("CONV_RELU_BACKWARD");
  tensor_t d_dtmp = context->d_dtmp;
  PINF("d_dy");
  // lcache_dump_stat(cache);

  AWNN_CHECK_EQ(S_OK, relu_backward_device(d_dtmp, cache, d_dy));
  AWNN_CHECK_EQ(S_OK, convolution_backward_device(handle, d_dx, d_dw, cache,
                                                  params, d_dtmp));

  return S_OK;
}

status_t conv_iden_relu_forward_device(cublasHandle_t handle,
                                       tensor_t const d_x,
                                       tensor_t const d_iden, tensor_t d_w,
                                       lcache_t* cache,
                                       conv_param_t const params, tensor_t d_y,
                                       struct layer_context_device* context) {
  AWNN_CHECK_EQ(d_x.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_iden.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_w.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_y.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(
      d_x.dim.dims[3],
      d_y.dim.dims[3]);  // in resnet tensor h/w doesn't change in each stage
  tensor_t d_tmp = context->d_tmp;
  AWNN_CHECK_EQ(
      S_OK, convolution_forward_device(handle, d_x, d_w, cache, params, d_tmp));

  elementwise_add_inplace_device<<<32, 1024>>>(d_tmp, d_iden);

  AWNN_CHECK_EQ(S_OK, relu_forward_device(d_tmp, cache, d_y));
  return S_OK;
}

status_t conv_iden_relu_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                        tensor_t d_diden, tensor_t d_dw,
                                        lcache_t* cache,
                                        conv_param_t const params,
                                        tensor_t const d_dy,
                                        struct layer_context_device* context) {
  AWNN_CHECK_EQ(d_dx.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_diden.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dw.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dy.mem_type, GPU_MEM);
  AWNN_NO_USE(context);  // backward doesn;t need temp

  AWNN_CHECK_EQ(S_OK, relu_backward_device(d_diden, cache, d_dy));
  AWNN_CHECK_EQ(S_OK, convolution_backward_device(handle, d_dx, d_dw, cache,
                                                  params, d_diden));

  return S_OK;
}

/** Create cache memory for this and all its childer layers */
void resblock_create_context_device(struct layer_context_device** ptr_context,
                                    tensor_t d_y) {
  // one context for this layer, the other two for its children
  uint nr_child_context = 2;
  *ptr_context = (struct layer_context_device*)mem_alloc(
      sizeof(struct layer_context_device) * (2 + nr_child_context));

  struct layer_context_device* context = *ptr_context;
  for (uint i = 0; i < nr_child_context + 2; i++) {
    context[i].d_tmp = tensor_make_alike_device(d_y);
    context[i].d_dtmp = tensor_make_alike_device(d_y);
  }
}

/** Free cache memory for this and all its childer layers */
void resblock_destroy_context_device(struct layer_context_device* context) {
  uint nr_child_context = 2;
  for (uint i = 0; i < nr_child_context + 2; i++) {
    layer_context_destroy_device(&context[i]);
  }
  mem_free(context);
}

status_t resblock_forward_device(cublasHandle_t handle, tensor_t const d_x,
                                 tensor_t d_w1, tensor_t d_w2, lcache_t* cache,
                                 conv_param_t const params, tensor_t d_y,
                                 struct layer_context_device* context) {
  tensor_t d_tmp = context[0].d_tmp;
  // TODO: pass context
  conv_relu_forward_device(handle, d_x, d_w1, cache, params, d_tmp,
                           &context[2]);
  conv_iden_relu_forward_device(handle, d_tmp, d_x, d_w2, cache, params, d_y,
                                &context[3]);

  return S_OK;
}
status_t resblock_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                  tensor_t d_dw1, tensor_t d_dw2,
                                  lcache_t* cache, conv_param_t const params,
                                  tensor_t const d_dy,
                                  struct layer_context_device* context) {
  tensor_t d_dtmp = context[0].d_dtmp;
  tensor_t d_dx_iden = context[1].d_dtmp;

  conv_iden_relu_backward_device(handle, d_dtmp, d_dx_iden, d_dw2, cache,
                                 params, d_dy, &context[3]);
  // TODO: pass context
  conv_relu_backward_device(handle, d_dx, d_dw1, cache, params, d_dtmp,
                            &context[2]);

  elementwise_add_inplace_device<<<32, 1024>>>(d_dx, d_dx_iden);

  return S_OK;
}
