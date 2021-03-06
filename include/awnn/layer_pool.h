/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/tensor.h"
#include "awnn/layer.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief forwarding for global_avg_pool
 *
 * @param x data input of shape (N, C, H, W)
 * @param cache [output] intermidate results
 * @param y [output] forwarding output (shape (N,C,1,1))
 *
 * @return S_OK if success, otherwise S_ERR or define your error type in common.h
 *
 * Note:
 *  * all tensor_t should be pre-allocated
 *  * cache will be populated by forward function.
 *
 * See https://fengggli.github.io/gpu-computing-materials/layers/pool.html 
 */
status_t global_avg_pool_forward(tensor_t const x, lcache_t *cache, tensor_t y);
status_t global_avg_pool_forward_device(tensor_t const x, lcache_t *cache, tensor_t y);

/*
 * @brief backprop
 *
 * @param dx [output] gradient w.r.t input
 * @param cache
 * @param dy gradient from upper layer
 *
 * @return S_OK if success, otherwise S_ERR or define your error type in common.h
 *
 * Note: all tensor_t should be pre-allocated
 */
status_t global_avg_pool_backward(tensor_t dx, lcache_t *cache, tensor_t const dy);
status_t global_avg_pool_backward_device(tensor_t dx, lcache_t *cache, tensor_t const dy);

void do_global_pool_forward(tensor_t x, tensor_t y);
void do_global_pool_backward(tensor_t dx, tensor_t dy);

void do_max_pool_forward(tensor_t x, tensor_t y, uint kernel_size);
void do_max_pool_backward(tensor_t dx, tensor_t dy, uint kernel_size,
                          tensor_t x, tensor_t y);

#ifdef __cplusplus
}
#endif


