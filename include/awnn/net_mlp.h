//
// Created by lifen on 3/25/19.
//

/*
 * Fully connected net
 */

#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  tensor_t *learnable_params_data;
  tensor_t *learnable_params_diff;
} model_t;

tensor_t *layer_output;

/*
 * @brief Create a fc model
 */
status_t fcnet_init(model_t *model, // output
    uint nr_input, //
    uint nr_hidden,
    uint nr_output);


/* Compute the scores for a batch or input*/
/* TODO: need better design.
status_t fcnet_forward(model_t const *model, tensor_t x, tensor_t y, T *loss);

/* Compute the loss and gradients(of the weights)for a batch or input*/
status_t fcnet_backward(model_t *model, tensor_t x, tensor_t const y, T *loss);

/* Compute the scores for a batch of input (inference only)*/
status_t fcnet_infer_only(model_t const *model, tensor_t x);

#ifdef __cplusplus
}
#endif

