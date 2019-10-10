#include "awnn/im2col.h"
#include "awnn/layer_conv.h"

#ifdef USE_OPENBLAS
#include "cblas.h"
#endif
#ifdef USE_MKL
#include "mkl.h"
#endif

#ifdef AWNN_USE_OPENMP
#include <omp.h>
#endif

void do_conv_forward_perimg(tensor_t const x, tensor_t const w, tensor_t y,
                            int pad, int stride) {
  uint N, C, H, W; /** input dims */
  uint F, HH, WW;  /** filter spatial*/
  uint Hout, Wout; /** output spatial dimension*/

  N = x.dim.dims[0];
  C = x.dim.dims[1];
  H = x.dim.dims[2];
  W = x.dim.dims[3];

  // weight has F, C HH, WW or Fx(C*HH*WW)
  F = w.dim.dims[0];
  HH = w.dim.dims[2];
  WW = w.dim.dims[3];


  // Hout = (H + 2 * pad - HH) / stride + 1; // total strides needed over rows
  Hout = y.dim.dims[2];  // total strides needed over rows
  Wout = y.dim.dims[3];  // total strides needed over cols

  uint bottom_dim = C * H * W;
  uint top_dim = F * Hout * Wout;

  T* col_buff = alloc_col_buffer(C, HH, WW, Hout, Wout);
  for (uint n = 0; n < N; n++) {  // for each img
    im2col_cpu(x.data + n * (bottom_dim), C, H, W, HH, WW, pad, stride,
               col_buff);  //((C*HH*WW)*(Hout*Wout)

    // y = w*x_cols;
    awnn_gemm(CblasNoTrans, CblasNoTrans, F, Hout * Wout, (C * HH * WW), 1.0,
              w.data, col_buff, 0.0, y.data + n * (top_dim));
  }
  free_col_buffer(col_buff);
}

status_t conv_forward_perimg(tensor_t const x, tensor_t const w,
                             lcache_t* cache, conv_param_t const params,
                             tensor_t y) {
  do_conv_forward_perimg(x, w, y, params.padding, params.stride);
  if (cache) {
    lcache_push(cache, x);
    lcache_push(cache, w);
  }
  return S_OK;
}

status_t do_conv_backward_perimg(tensor_t dx, tensor_t dw, tensor_t const dy,
                                 tensor_t x, tensor_t w, int pad, int stride) {
  uint N, C, H, W; /** input dims */
  uint F, HH, WW;  /** filter spatial*/
  uint Hout, Wout; /** output spatial dimension*/

  N = dx.dim.dims[0];
  C = dx.dim.dims[1];
  H = dx.dim.dims[2];
  W = dx.dim.dims[3];

  // weight has F, C HH, WW or Fx(C*HH*WW)
  F = dw.dim.dims[0];
  HH = dw.dim.dims[2];
  WW = dw.dim.dims[3];


  // Hout = (H + 2 * pad - HH) / stride + 1; // total strides needed over rows
  Hout = dy.dim.dims[2];  // total strides needed over rows
  Wout = dy.dim.dims[3];  // total strides needed over cols

  uint bottom_dim = C * H * W;
  uint top_dim = F * Hout * Wout;

  uint col_buffer_size = (C * HH * WW) * (Hout * Wout);

  // clear dw
  tensor_fill_scalar(dw, 0);

  T* col_buff = alloc_col_buffer(C, HH, WW, Hout, Wout);

  for (uint n = 0; n < N; n++) {  // for each img
    im2col_cpu(x.data + n * (bottom_dim), C, H, W, HH, WW, pad, stride,
               col_buff);  //((C*HH*WW)*(Hout*Wout)

    // gradient w.r.t weight: dw = dy* T(x_col)
    awnn_gemm(CblasNoTrans, CblasTrans, F, (C * HH * WW), Hout * Wout, 1.0,
              dy.data + n * (top_dim), col_buff, 1.0, dw.data);
    // gradient w.r.t input: dx_col = T(w)*dy
    awnn_gemm(CblasTrans, CblasNoTrans, (C * HH * WW), Hout * Wout, F, 1.0,
              w.data, dy.data + n * (top_dim), 0.0, col_buff);
    col2im_cpu(col_buff, C, H, W, HH, WW, pad, stride,
               dx.data + n * (bottom_dim));  //((C*HH*WW)*(Hout*Wout)
  }

  free_col_buffer(col_buff);
  return S_OK;
}

status_t conv_backward_perimg(tensor_t dx, tensor_t dw, lcache_t* cache,
                              conv_param_t const params, tensor_t const dy) {
  tensor_t w = lcache_pop(cache);
  tensor_t x = lcache_pop(cache);

  int pad = params.padding;
  int stride = params.stride;
  return do_conv_backward_perimg(dx, dw, dy, x, w, pad, stride);
}
