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
} model_data_t;

typedef struct {
  model_data_t *model_data;
  uint max_batch_sz;
  T reg;
} model_t;



tensor_t *layer_output;

/*
 * @brief Create a mlp model
 *
 * This will allocate all space for:
 *  1. weight/bias for each hidden layer
 * TODO: allocate space for cache
 */
status_t mlp_init(model_t *model, // output
    uint max_batch_sz,
    uint input_dim, //
    uint nr_hidden_layers,
    uint hidden_dim,
    uint output_dim,
    T reg);

/*
 * @brief Destroy a mlp model
 */
status_t mlp_finalize(model_t *model);


/* Compute the scores for a batch or input, infer only*/
status_t mlp_scores(model_t const *model, tensor_t x, tensor_t scores);

/* Compute loss for a batch of (x,y), do forward/backward, and update gradients*/
status_t mlp_loss(model_t const *model, tensor_t x, label_t const labels[], T * loss);

#ifdef __cplusplus
}
#endif

