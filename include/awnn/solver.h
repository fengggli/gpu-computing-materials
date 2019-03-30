//
// Created by lifen on 3/25/19.
//
#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"
#include "awnn/data_utils.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
}solver_handle_t;

enum OptimizeMethod { OPTIM_SGD = 0, OPTIM_SGD_MOMENTUM = 1, OPTIM_NODEF = -1 };

typedef struct{
  uint batch_size;
  OptimizeMethod optimize_method = OPTIM_SGD;
  T learning_rate;
} solver_config_t;

/* Solver does the following:
 *  1. load validation data
 *  2. for each minibatch i:
 *    * load a batch(x,y) from training
 *    * compute model forward + backward, get all gradients
 *    * update model weights
 *  3. It also print progress info and statistics info (error, etc)
 */
status_t solver_init(solver_handle_t *handle, data_t data, solver_config_t config);
status_t solver_train(solver_handle_t const *handle);

#ifdef __cplusplus
}
#endif
