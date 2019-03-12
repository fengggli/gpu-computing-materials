/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_CONV_H_
#define LAYER_CONV_H_

#include "tensor.h"

typedef struct{
  int stride;
  int padding;
} conv_param_t;

/*
 * @brief forwarding for conv2d
 *
 * @param x data input of shape (N, C, H, W)
 * @param w filters of shape (F, C, HH, WW)
 * @param cache [output] intermidate results
 * @param y [output] forwarding output
 *
 * Note: all tensor_t should be pre-allocated

 *
 * See conv_forward_naive https://github.com/fengggli/cs231n-assignments/blob/master/assignment2/cs231n/layers.py
 */
void convolution_forward(tensor_t x, tensor_t w, tensor_t* cache, conv_param_t params, tensor_t y);

/*
 * @brief backprop
 *
 * @param dx [output] gradient w.r.t input
 * @param dw [output] gradient w.r.t filters
 * @param cache
 * @param dy gradient from upper layer
 */
void convolution_backward(tensor_t dx, tensor_t dw, tensor_t *cache, tensor_t dy);

#endif