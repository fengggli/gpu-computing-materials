/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#pragma once
#include "awnn/layer.h"

#ifdef __cplusplus
extern "C" {
#endif

status_t layer_relu_forward(tensor_t const x, lcache_t *cache, tensor_t y);
status_t layer_relu_backward(tensor_t dx, lcache_t *cache, tensor_t const dy);

#ifdef __cplusplus
}
#endif
