#include "awnn/net_mlp.h"
#include "awnn/solver.h"

// TODO: Need data input;
status_t solver_init(solver_handle_t *ptr_handle, model_t *ptr_model,
                     data_t *ptr_data, solver_config_t *ptr_config) {
  ptr_handle->model = *ptr_model;
  ptr_handle->data = *ptr_data;
  ptr_handle->config = *ptr_config;
  return S_OK;
}

status_t solver_train(solver_handle_t const *ptr_solver) {
  // TODO: need a data loading util
  // Compute loss and gradient
  model_t model = ptr_solver->model;
  T loss = 0;
  uint x_shape[] = {model.max_batch_sz, model.input_dim};
  tensor_t x = tensor_make_linspace(-0.1, 0.5, x_shape, 2);
  label_t *labels = label_make_random(model.max_batch_sz, model.output_dim);

  mlp_loss(&model, x, labels, &loss);

  // optimizer
  param_t *p_param;
  // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
  int i = 0;  // TODO: some gabage interted in the end of paramlist.
  list_for_each_entry(p_param, model.list_all_params, list) {
    PINF("updating %s...", p_param->name);
    tensor_t param = p_param->data;
    tensor_t dparam = p_param->diff;

    // sgd
    T *pelem;
    uint ii;
    T learning_rate = ptr_solver->config.learning_rate;

    AWNN_CHECK_GT(learning_rate, 0);
    tensor_for_each_entry(pelem, ii, dparam) { (*pelem) *= learning_rate; }
    tensor_elemwise_op_inplace(param, dparam, TENSOR_OP_SUB);
    PINF("updating %s complete.", p_param->name);

    i++;
    if (i >= 4) break;
  }

  PINF("clean up.");
  tensor_destroy(&x);
  label_destroy(labels);
}
