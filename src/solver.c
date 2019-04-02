#include "awnn/net_mlp.h"
#include "awnn/solver.h"

// TODO: Need data input;
status_t solver_init(solver_handle_t *ptr_handle, model_t *ptr_model,
                     data_t *ptr_data, solver_config_t *ptr_config) {
  ptr_handle->model = ptr_model;
  ptr_handle->data = ptr_data;
  ptr_handle->config = ptr_config;
  return S_OK;
}

status_t solver_train(solver_handle_t const *ptr_solver) {
  // TODO: need a data loading util
  //Compute loss and gradient
  model_t *ptr_model = ptr_solver->model;
  T loss = 0;
  uint x_shape[] = {ptr_model->max_batch_sz, ptr_model->input_dim};
  tensor_t x = tensor_make_linspace(-0.1, 0.5, x_shape, 2);
  label_t *labels = label_make_random(ptr_model->max_batch_sz, ptr_model->output_dim);

  mlp_loss(&ptr_model, x, labels, &loss);

  // optimizer

  tensor_destroy(x);
  label_destroy(labels);

  loss, grads = self.model.loss(X_batch, y_batch)
  self.loss_history.append(loss)

  //Perform a parameter update
  for p, w in self.model.params.items():
  dw = grads[p]
  config = self.optim_configs[p]
  next_w, next_config = self.update_rule(w, dw, config)
  self.model.params[p] = next_w
  self.optim_configs[p] = next_config

}

