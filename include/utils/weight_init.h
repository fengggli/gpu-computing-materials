/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef WEIGHT_INIT_H_
#define WEIGHT_INIT_H_

#include "awnn/common.h"
#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Init a tensor with linspace*/
status_t weight_init_linspace(tensor_t t, T start, T stop);

/* Init a fc weight layer with kaiming*/
status_t weight_init_fc_kaiming(tensor_t weight, tensor_t bias);

/* Init a fc weight layer, (0, scale) and 0*/
status_t weight_init_fc(tensor_t weight, tensor_t bias, T weight_scale);

/* Init a tensor with normal distribution*/
status_t weight_init_norm(tensor_t t, T mean, T std);

/* Init a tensor with normal distribution*/
status_t weight_init_zeros(tensor_t t);

/* Init a tensor using kaiming normalization */
status_t weight_init_kaiming(tensor_t t);

#ifdef __cplusplus
}
#endif
#endif
