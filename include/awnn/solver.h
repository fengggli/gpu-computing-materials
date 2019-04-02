//
// Created by lifen on 3/25/19.
//
#pragma once

#include "awnn/common.h"
#include "awnn/data_utils.h"
#include "awnn/net_mlp.h"
#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OptimizeMethod { OPTIM_SGD = 0, OPTIM_SGD_MOMENTUM = 1, OPTIM_NODEF = -1 } opt_method_t;

typedef struct {
  uint batch_size;
  opt_method_t optimize_method;
  T learning_rate;
} solver_config_t;

typedef struct {
  data_t *data;
  solver_config_t *config;
  model_t *model;
} solver_handle_t;

/* Solver does the following:
 *  1. load validation data
 *  2. for each minibatch i:
 *    * load a batch(x,y) from training
 *    * compute model forward + backward, get all gradients
 *    * update model weights
 *  3. It also print progress info and statistics info (error, etc)
 */
status_t solver_init(solver_handle_t *ptr_handle, model_t *ptr_model,
                     data_t *ptr_data, solver_config_t *ptr_config);

// Iterate over all minibatches and update gradient correspondingly
status_t solver_train(solver_handle_t const *handle);

#ifdef __cplusplus
}
#endif
