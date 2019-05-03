#include "awnn/common.h"

#include "awnndevice/cublas_wrappers.cuh"
#include "awnndevice/device_utils.cuh"
#include "awnndevice/layer_sandwich_device.cuh"
#include "awnndevice/layer_conv_device.cuh"

__global__ void do_device_relu_forward(tensor_t d_x, tensor_t d_y){
  for (uint i : grid_stride_range(0u, d_capacity(d_x))) {
    d_y.data[i] = d_x.data[i] > 0 ? d_x.data[i] : 0.0;
  }
}
__global__ void do_device_relu_backward(tensor_t d_dx, tensor_t d_x, tensor_t d_dy){

  for (uint i : grid_stride_range(0u, d_capacity(d_x))) {
    d_dx.data[i] = d_x.data[i] > 0 ? d_dy.data[i] : 0.0;
  }
}

status_t relu_forward_device(tensor_t const d_x,
                                  lcache_t* cache,
                                  tensor_t d_y) {
  do_device_relu_forward<<<32, 1024>>>(d_x, d_y);

  if(cache){
    lcache_push(cache, d_x);
  }

  // print_tensor_device<<<1,1>>>(d_x);

  return S_OK;
}
status_t relu_backward_device(tensor_t const d_dx,
                                  lcache_t* cache,
                                  tensor_t d_dy) {
  tensor_t d_x = lcache_pop(cache);
  PINF("DEVICE");
/*  print_tensor_device<<<1,1>>>(d_dy);*/
  /*print_tensor_device<<<1,1>>>(d_x);*/
  /*print_tensor_device<<<1,1>>>(d_dx);*/
  do_device_relu_backward<<<32, 1024>>>(d_dx, d_x, d_dy);
/*  print_tensor_device<<<1,1>>>(d_dy);*/
  /*print_tensor_device<<<1,1>>>(d_x);*/
  /*print_tensor_device<<<1,1>>>(d_dx);*/
  PINF("DEVICE");

  return S_OK;
}


status_t conv_relu_forward_device(cublasHandle_t handle, tensor_t const d_x,
                                  tensor_t d_w, lcache_t* cache,
                                  conv_param_t const params, tensor_t d_y) {
  AWNN_CHECK_EQ(d_x.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_w.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_y.mem_type, GPU_MEM);
  status_t ret = S_ERR;
  tensor_t d_tmp = tensor_make_alike_device(d_y);

  AWNN_CHECK_EQ(S_OK, convolution_forward_device(handle, d_x, d_w, cache, params, d_tmp));
  AWNN_CHECK_EQ(S_OK, relu_forward_device(d_tmp, cache, d_y));

  tensor_destroy_device(&d_tmp);

  return S_OK;
}

status_t conv_relu_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                   tensor_t d_dw, lcache_t* cache,
                                   conv_param_t const params,
                                   tensor_t const d_dy) {
  AWNN_CHECK_EQ(d_dx.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dw.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dy.mem_type, GPU_MEM);
  status_t ret = S_ERR;

  tensor_t d_tmp = tensor_make_alike_device(d_dy);

  AWNN_CHECK_EQ(S_OK, relu_backward_device(d_tmp, cache, d_dy));
  AWNN_CHECK_EQ(S_OK, convolution_backward_device(handle, d_dx, d_dw, cache, params, d_tmp));

  tensor_destroy_device(&d_tmp);
  ret = S_OK;
  return ret;
}

status_t conv_iden_relu_forward_device(
    cublasHandle_t handle, tensor_t const d_x, tensor_t const d_iden,
    tensor_t d_w, lcache_t* cache, conv_param_t const params, tensor_t d_y) {
  AWNN_CHECK_EQ(d_x.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_iden.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_w.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_y.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(
      d_x.dim.dims[3],
      d_y.dim.dims[3]);  // in resnet tensor h/w doesn't change in each stage
  tensor_t d_tmp = tensor_make_alike_device(d_y);
  AWNN_CHECK_EQ(S_OK, convolution_forward_device(handle, d_x, d_w, cache, params, d_tmp));


  elementwise_add_inplace_device<<<32,1024>>>(d_tmp, d_iden);

  // tensor_elemwise_op_inplace(tmp, iden, TENSOR_OP_ADD);

  AWNN_CHECK_EQ(S_OK, relu_forward_device(d_tmp, cache, d_y));
  tensor_destroy_device(&d_tmp);
  return S_OK;
}

status_t conv_iden_relu_backward_device(cublasHandle_t handle, tensor_t d_dx,
                                        tensor_t d_diden, tensor_t d_dw,
                                        lcache_t* cache,
                                        conv_param_t const params,
                                        tensor_t const d_dy) {
  AWNN_CHECK_EQ(d_dx.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_diden.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dw.mem_type, GPU_MEM);
  AWNN_CHECK_EQ(d_dy.mem_type, GPU_MEM);


  AWNN_CHECK_EQ(S_OK, relu_backward_device(d_diden, cache, d_dy));
  AWNN_CHECK_EQ(S_OK, convolution_backward_device(handle, d_dx, d_dw, cache, params, d_diden));

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


  elementwise_add_inplace_device<<<32, 1024>>>(d_dx, d_dx_iden);
  // tensor_elemwise_op_inplace_device(d_dx, d_dx_iden, TENSOR_OP_ADD);
  tensor_destroy_device(&d_tmp);
  tensor_destroy_device(&d_dx_iden);

  return S_OK;
}
