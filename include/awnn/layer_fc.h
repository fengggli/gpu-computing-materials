/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_FC_H_
#define LAYER_FC_H_

#include "awnn/tensor.h"

/*
 * @brief Fully connected layer implementation
 *
 * @param x data input of shape (N, d1, d2,...); D = d1*d2*...dn
 * @param w weight shape (D, M)
 * @param b  shape (M,)
 * @param cache intermidate results
 * @param y forwarding output
 *
 * Note: all tensor_t should be pre-allocated
 */
void layer_fc_forward(tensor_t x, tensor_t w, tensor_t b, tensor_t *cache, tensor_t y);

/*
 * @brief Fully connected layer backprop
 * 
 * Note: all tensor_t should be pre-allocated
 */
void layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t b,tensor_t *cache, tensor_t dy);

#endif
