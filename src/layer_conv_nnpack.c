#include "awnn/layer_conv.h"
#include "nnpack.h"
#include "nnpack/reference.h"
#include "pthreadpool.h"

#include <awnn/memory.h>
#include <printf.h>

static int nnpack_initialized = 0;

status_t convolution_forward_nnpack(conv_method_t algo, tensor_t const x,
                                    tensor_t const w, lcache_t* cache, conv_param_t const params,
                                    tensor_t y) {
#ifndef AWNN_USE_FLT32
  PERR("nnpack doesn's support double");
  return S_ERR;
#else

  enum nnp_status status;
  uint batch_size = x.dim.dims[0];
  uint input_channel = x.dim.dims[1];
  uint output_channel = w.dim.dims[0];
  struct nnp_size input_size, kernel_size;
  input_size.height = x.dim.dims[2];
  input_size.width = x.dim.dims[3];
  /*output_size.height = y.dim.dims[2];*/
  /*output_size.width = y.dim.dims[3];*/
  struct nnp_padding pad;
  pad.bottom = params.padding;
  pad.top = params.padding;
  pad.right = params.padding;
  pad.left = params.padding;

  kernel_size.height = w.dim.dims[2];
  kernel_size.width = w.dim.dims[3];

  pthreadpool_t thrd_pool = NULL;
  struct nnp_profile *profile = NULL;

  const float *input = x.data;
  const float *kernel = w.data;
  uint bias_shape[] = {output_channel};
  tensor_t t_bias = tensor_make_zeros(bias_shape, 1);
  const float *bias = t_bias.data;
  float *output = y.data;

  if (!nnpack_initialized) {
    status = nnp_initialize();
    AWNN_CHECK_EQ(status, nnp_status_success);
    nnpack_initialized = 1;
  }

  // allocate mem space
  // TODO: this can be in the first iteration
#if 0
  size_t scratch_size = 0;
  status = nnp_convolution_output(algo, batch_size,
                                  input_channel, output_channel, input_size,
                                  pad, kernel_size, NULL, NULL, NULL, NULL,
                                  NULL, &scratch_size, nnp_activation_identity,
                                  NULL,  // activation param
                                  thrd_pool, profile);
  AWNN_CHECK_EQ(status, nnp_status_success);

  // TODO: this must be 64 aligned, see nnpack.h
  float *scratch_mem = mem_alloc(scratch_size);

  status = nnp_convolution_output(
      algo, batch_size, input_channel,
      output_channel, input_size, pad, kernel_size, input, kernel, bias,
      output, scratch_mem, &scratch_size, nnp_activation_identity,
      NULL,  // activation param
      thrd_pool, profile);
#else
  if (algo != CONV_METHOD_NNPACK_REF) {
    status =
      nnp_convolution_output(algo, batch_size, input_channel, output_channel,
                             input_size, pad, kernel_size, input, kernel, bias,
                             output, NULL, NULL, nnp_activation_identity,
                             NULL,  // activation param
                             thrd_pool, profile);
    AWNN_CHECK_EQ(status, nnp_status_success);
  } else {
    struct nnp_size output_subsampling;
    output_subsampling.height = 1;
    output_subsampling.width = 1;

    nnp_convolution_output__reference(
        batch_size, input_channel, output_channel, input_size, pad, kernel_size,
        output_subsampling, input, kernel, bias, output, thrd_pool);
  }
#endif

  // shadow copy
  tensor_t cached_x_shadow = x;
  tensor_t cached_w_shadow = w;

  // TODO put w and data
  if (cache) {
    lcache_push(cache, cached_x_shadow);
    lcache_push(cache, cached_w_shadow);
  }
  tensor_destroy(&t_bias);

  return S_OK;
#endif
}

status_t convolution_backward_nnpack(conv_method_t algo, tensor_t dx,
                                     tensor_t dw, lcache_t* cache,
                                     conv_param_t const params,
                                     tensor_t const dout) {
  tensor_t x, w;

  // NOTE : the order of pop matters, should be flattened_x, w, x (reverse of
  // forward)
  w = lcache_pop(cache);
  x = lcache_pop(cache);

  enum nnp_status status;
  uint batch_size = x.dim.dims[0];
  uint input_channel = x.dim.dims[1];
  uint output_channel = w.dim.dims[0];
  struct nnp_size input_size, kernel_size;
  input_size.height = x.dim.dims[2];
  input_size.width = x.dim.dims[3];
  /*output_size.height = y.dim.dims[2];*/
  /*output_size.width = y.dim.dims[3];*/
  struct nnp_padding pad;
  pad.bottom = params.padding;
  pad.top = params.padding;
  pad.right = params.padding;
  pad.left = params.padding;

  kernel_size.height = w.dim.dims[2];
  kernel_size.width = w.dim.dims[3];

  pthreadpool_t thrd_pool = NULL;
  struct nnp_profile* profile = NULL;

  // How about bias?
  // input gradient
  if (algo != CONV_METHOD_NNPACK_REF) {
    status = nnp_convolution_input_gradient(
        algo, batch_size, input_channel, output_channel, input_size, pad,
        kernel_size, dout.data, w.data, dx.data,
        NULL,  // let it allocate scratch on the fly
        NULL, nnp_activation_identity, NULL,  // activation and its param
        thrd_pool, profile);                  // thread pool and profile
    AWNN_CHECK_EQ(nnp_status_success, status);

    // w gradient
    status = nnp_convolution_kernel_gradient(
        algo, batch_size, input_channel, output_channel, input_size, pad,
        kernel_size, x.data, dout.data, dw.data,
        NULL,  // let it allocate scratch on the fly
        NULL, nnp_activation_identity, NULL,  // activation and its param
        thrd_pool, profile);                  // thread pool and profile
    AWNN_CHECK_EQ(nnp_status_success, status);
  } else {
    nnp_convolution_input_gradient__reference(
        batch_size, input_channel, output_channel, input_size, pad, kernel_size,
        dout.data, w.data, dx.data, thrd_pool);
    nnp_convolution_kernel_gradient__reference(
        batch_size, input_channel, output_channel, input_size, pad, kernel_size,
        x.data, dout.data, dw.data, thrd_pool);
  }

  return S_OK;
}
