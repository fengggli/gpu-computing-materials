/*
 * Resnet
 */
#include "awnn/net_resnet.h"

status_t resnet_init(
    model_t *model,  // output
    uint max_batch_sz,
    uint input_dim[3],  // C,H,W
    uint output_dim,    // nr_classes
    uint nr_stages,
    uint nr_blocks[MAX_STAGES],  // how many residual blocks in each stage
    T reg, normalize_method_t normalize_method) {
  return S_ERR;
}

/*
 * @brief Destroy a mlp model
 */
status_t resnet_finalize(model_t *model) { return S_ERR; }

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
