/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef NET_RESNET_H_
#define NET_RESNET_H_

#include "awnn/net.h"

#include "awnn/common.h"
#include "awnn/tensor.h"
#include "utils/list.h"

#define MAX_STAGES (5)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NORMALIZE_NONE = 0,
  NORMALIZE_BATCH = 1,
} normalize_method_t;

typedef struct model {
  dim_t input_dim;
  int output_dim;
  int nr_stages;
  int nr_blocks[MAX_STAGES];
  T reg;

  struct list_head list_all_params[1];   // list of all learnable params
  struct list_head list_layer_out[1];    // list of output of each layer
  struct list_head list_layer_in[1];     // list of input of each layer
  struct list_head list_layer_cache[1];  // list of layer cache.
} model_t;

status_t resnet_init(
    model_t *model,  // output
    dim_t input_dim,
    int output_dim,  // nr_classes
    int nr_stages,
    int nr_blocks[MAX_STAGES],  // how many residual blocks in each stage
    T reg, normalize_method_t normalize_method);

/*
 * @brief Destroy a mlp model
 */
status_t resnet_finalize(model_t *model);

/* Compute the scores for a batch or input, infer only*/
tensor_t resnet_forward_infer(model_t const *model, tensor_t x);

/* Compute the scores for a batch or input, update cache*/
tensor_t resnet_forward(model_t const *model, tensor_t x);

/* Compute loss for a batch of (x,y), do forward/backward, and update
 * gradients*/
status_t resnet_loss(model_t const *model, tensor_t x, label_t const labels[],
                     T *ptr_loss);

#ifdef __cplusplus
}
#endif
#endif
