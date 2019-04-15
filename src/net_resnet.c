/*
 * Resnet
 */
#include "awnn/net_resnet.h"

status_t resnet_init(
    model_t *model,   // output
    dim_t input_dim,  // NCHW
    uint output_dim,  // nr_classes
    uint nr_stages,
    uint nr_blocks[MAX_STAGES],  // how many residual blocks in each stage
    T reg, normalize_method_t normalize_method) {
  // save this configuration
  model->input_dim = input_dim;
  model->output_dim = output_dim;

  model->nr_stages = nr_stages;
  model->reg = reg;
  for (uint i = 0; i < MAX_STAGES; ++i) {
    if (i < nr_stages)
      model->nr_blocks[i] = nr_blocks[i];
    else
      model->nr_blocks[i] = 0;
  }

  // init all list structure
  init_list_head(model->list_all_params);
  init_list_head(model->list_layer_out);
  init_list_head(model->list_layer_in);
  init_list_head(model->list_layer_cache);

  return S_OK;
}

/*
 * @brief Destroy a mlp model
 */
status_t resnet_finalize(model_t *model) {
  net_free_cache(model->list_layer_cache);
  net_free_params(model->list_layer_out);
  // net_free_params(model->list_layer_in); // TODO: fix the double free here.
  net_free_params(model->list_all_params);
  return S_OK;
}

/* Compute the scores for a batch or input, infer only*/
status_t resnet_forward_infer(model_t const *model, tensor_t x) {
  return S_ERR;
}

/* Compute the scores for a batch or input, update cache*/
status_t resnet_forward(model_t const *model, tensor_t x) { return S_ERR; }

/* Compute loss for a batch of (x,y), do forward/backward, and update
 * gradients*/
status_t resnet_loss(model_t const *model, tensor_t x, label_t const labels[],
                     T *ptr_loss) {
  return S_ERR;
}
