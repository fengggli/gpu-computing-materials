//
// Created by lifen on 3/25/19.
//
#pragma once

#include "awnn/common.h"
#include "awnn/data_utils.h"
#include "awnn/net.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void sgd_update(param_t * p_param, T learning_rate){
  tensor_t param = p_param->data;
  tensor_t dparam = p_param->diff;
  T *pelem;
  uint ii;
  AWNN_CHECK_GT(learning_rate, 0);

  tensor_for_each_entry(pelem, ii, dparam) { (*pelem) *= learning_rate; }
  tensor_elemwise_op_inplace(param, dparam, TENSOR_OP_SUB);
  PDBG("updating %s complete.", p_param->name);
}

#ifdef __cplusplus
}
#endif
