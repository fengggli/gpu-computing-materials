/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_CONV_H_
#define LAYER_CONV_H_


#include "awnn/layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  int stride;
  int padding;
} conv_param_t;


/*
 * @brief forwarding for conv2d
 *
 * @param x data input of shape (N, C, H, W)
 * @param w filters, shape (F, C, HH, WW)
 * @param cache [output] intermidate results
 * @param y [output] forwarding output, shape (N,C,HH,WW)
 *
 * @return S_OK if success, otherwise S_ERR or define your error type in common.h
 *
 * Note:
 *  * all tensor_t should be pre-allocated
 *  * cache will be populated by forward function.
 *
 * See conv_forward_naive https://github.com/fengggli/cs231n-assignments/blob/master/assignment2/cs231n/layers.py
 */
status_t convolution_forward(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y);

/*
 * @brief backprop
 *
 * @param dx [output] gradient w.r.t input
 * @param dw [output] gradient w.r.t filters
 * @param cache
 * @param dy gradient from upper layer
 *
 * @return S_OK if success, otherwise S_ERR or define your error type in common.h
 */
status_t convolution_backward(tensor_t dx, tensor_t dw, lcache_t const *cache, tensor_t const dy);

#ifdef __cplusplus
}
#endif

#endif
