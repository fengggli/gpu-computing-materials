//
// Created by lifen on 3/25/19.
//
#pragma once

#include "awnn/common.h"
#include "awnn/data_utils.h"
// #include "awnn/net_mlp.h"
typedef struct model model_t;
typedef struct param param_t;

#ifdef __cplusplus
extern "C" {
#endif

void sgd_update(param_t *p_param, T learning_rate);
void sgd_update_momentum(param_t *p_param, T learning_rate, T momentum);

double check_val_accuracy(data_loader_t *loader, int val_sz, int batch_sz,
                          model_t const *model,
                          tensor_t (*func_forward_infer)(model_t const *,
                                                         tensor_t));
double check_train_accuracy(data_loader_t *loader, int sample_sz,
                            int batch_sz, model_t const *model,
                            tensor_t (*func_forward_infer)(model_t const *,
                                                           tensor_t));

#ifdef __cplusplus
}
#endif
