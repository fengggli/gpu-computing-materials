/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#pragma once

#include "awnn/layer.h"
#include "awnn/layer_conv.h"
#include "awnn/layer_fc.h"
#include "awnn/layer_relu.h"

#ifdef __cplusplus
extern "C" {
#endif

status_t fc_relu_forward(tensor_t const x, tensor_t const w, tensor_t b,
                         lcache_t *cache, tensor_t y);

status_t fc_relu_backward(tensor_t dx, tensor_t dw, tensor_t db,
                          lcache_t *cache, tensor_t const dy);

status_t conv_relu_forward(tensor_t const x, tensor_t const w, lcache_t *cache,
                           conv_param_t const params, tensor_t y);

status_t conv_relu_backward(tensor_t dx, tensor_t dw, lcache_t *cache,
                            conv_param_t const params, tensor_t const dout);

// TODO conv_bn_relu forward/backward
#ifdef __cplusplus
}
#endif
