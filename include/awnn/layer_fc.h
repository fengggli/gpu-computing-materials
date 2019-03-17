/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/layer.h"

/*
 * @brief Fully connected layer implementation
 *
 * @param x data input of shape (N, d1, d2,...); D = d1*d2*...dn
 * @param w weight shape (D, M)
 * @param b  shape (M,)
 * @param cache intermidate results
 * @param y forwarding output
 *
 * @return S_OK if success, otherwise S_ERR or define your error type in common.h
 *
 * Note: all tensor_t should be pre-allocated
 */
status_t layer_fc_forward(tensor_t x, tensor_t w, tensor_t b, lcache_t* cache, tensor_t y);

/*
 * @brief Fully connected layer backprop
 * 
 * @return S_OK if success, otherwise S_ERR or define your error type in common.h
 * Note: all tensor_t should be pre-allocated
 */
status_t layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t b, lcache_t const * cache, tensor_t dy);

