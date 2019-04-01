/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_SANDWICH_H_
#define LAYER_SANDWICH_H_
#include "awnn/layer.h"
#ifdef __cplusplus
extern "C" {
#endif

status_t layer_fc_relu_forward(tensor_t const x, tensor_t const w, tensor_t b,
                               lcache_t *cache, tensor_t y);

status_t layer_fc_relu_backward(tensor_t dx, tensor_t dw, tensor_t db,
                                lcache_t *cache, tensor_t const dy);

#ifdef __cplusplus
}
#endif
#endif
