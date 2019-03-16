/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_POOL_H_
#define LAYER_POOL_H_

#include "awnn/layer.h"

/*
 * @brief forwarding for global_avg_pool
 *
 * @param x data input of shape (N, C, H, W)
 * @param cache [output] intermidate results
 * @param y [output] forwarding output (shape (N,C,1,1))
 *
 * Note:
 *  * all tensor_t should be pre-allocated
 *  * cache will be populated by forward function.
 *
 * See https://fengggli.github.io/gpu-computing-materials/layers/pool.html 
 */
void global_avg_pool_forward(tensor_t x, lcache_t *cache, tensor_t y);

/*
 * @brief backprop
 *
 * @param dx [output] gradient w.r.t input
 * @param cache
 * @param dy gradient from upper layer
 *
 * Note: all tensor_t should be pre-allocated
 */
void global_avg_pool_backward(tensor_t dx, lcache_t const *cache, tensor_t dy);



#endif
