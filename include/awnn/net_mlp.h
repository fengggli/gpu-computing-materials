//
// Created by lifen on 3/25/19.
//

/*
 * Fully connected net
 */

#pragma once

#include "awnn/net.h"

#include "awnn/common.h"
#include "awnn/tensor.h"
#include "utils/list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct model {
  uint input_dim;
  uint output_dim;
  uint nr_hidden_layers;
  uint hidden_dims[MAX_DIM];
  uint max_batch_sz;
  T reg;

  struct list_head list_all_params[1];   // list of all learnable params
  struct list_head list_layer_out[1];    // list of output of each layer
  struct list_head list_layer_in[1];     // list of input of each layer
  struct list_head list_layer_cache[1];  // list of layer cache.
} model_t;

/*
 * @brief Create a mlp model
 *
 * This will allocate all space for:
 *  1. weight/bias for each hidden layer
 * TODO: allocate space for cache
 */
status_t mlp_init(model_t *model,  // output
                  uint max_batch_sz,
                  uint input_dim,  //
                  uint output_dim, uint nr_hidden_layers, uint hidden_dim[],
                  T reg);

/*
 * @brief Destroy a mlp model
 */
status_t mlp_finalize(model_t *model);

/* Compute the scores for a batch or input, infer only*/
tensor_t mlp_forward_infer(model_t const *model, tensor_t x);

/* Compute the scores for a batch or input, update cache*/
tensor_t mlp_forward(model_t const *model, tensor_t x);

/* Compute loss for a batch of (x,y), do forward/backward, and update
 * gradients*/
status_t mlp_loss(model_t const *model, tensor_t x, label_t const labels[],
                  T *ptr_loss);

#ifdef __cplusplus
}
#endif
