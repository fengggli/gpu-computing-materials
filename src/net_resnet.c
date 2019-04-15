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

  // copy of the output tensors of bottom layer
  tensor_t in, din;
  tensor_t out, dout;
  char in_name[MAX_STR_LENGTH];
  char out_name[MAX_STR_LENGTH];
  char cache_name[MAX_STR_LENGTH];

  uint N = input_dim.dims[0];
  uint C = input_dim.dims[1];  // 3.
  uint H = input_dim.dims[2];

  /*
   * I.  preparation
   */
  // layer in
  in = tensor_make_placeholder(input_dim.dims, 4);
  din = tensor_make(input_dim.dims, 4);
  snprintf(in_name, MAX_STR_LENGTH, "conv1.in");
  net_attach_param(model->list_layer_in, in_name, in, din);

  // prepare and init weights
  uint w_shape[] = {16, C, 3, 3};
  tensor_t w = tensor_make(w_shape, 4);
  tensor_t dw = tensor_make(w_shape, 4);
  char w_name[MAX_STR_LENGTH];
  snprintf(w_name, MAX_STR_LENGTH, "conv1.weight");
  net_attach_param(model->list_all_params, w_name, w, dw);
  weight_init_kaiming(w);  // TODO: init

  // layer out
  uint out_shape[] = {N, 16, 3, 3};
  out = tensor_make(out_shape, 4);
  dout = tensor_make(out_shape, 4);
  snprintf(out_name, MAX_STR_LENGTH, "conv1.out");
  net_attach_param(model->list_layer_out, out_name, out, dout);

  // cache
  snprintf(cache_name, MAX_STR_LENGTH, "conv1.cache");
  net_attach_cache(model->list_layer_cache, cache_name);

  C = 16;

  /*
   * II.  main stage
   */
  for (uint i_stage = 1; i_stage <= nr_stages; i_stage++) {
    for (uint i_blk = 0; i_blk < nr_blocks[i_stage - 1]; i_blk++) {
      char prefix[MAX_STR_LENGTH];
      snprintf(prefix, MAX_STR_LENGTH, "layer%u.%u", i_stage, i_blk);

      // In
      in = out;
      din = dout;
      snprintf(in_name, MAX_STR_LENGTH, "%s.in", prefix);
      net_attach_param(model->list_layer_in, in_name, in, din);

      // weight
      uint w_shape[] = {16, C, 3,3};

      tensor_t w1 = tensor_make(w_shape, 4);
      tensor_t dw1 = tensor_make_alike(w1);
      char w1_name[MAX_STR_LENGTH]; 
      snprintf(w1_name, MAX_STR_LENGTH, "%s.conv1.weight", prefix);
      net_attach_param(model->list_all_params, w1_name, w1, dw1);

      tensor_t w2 = tensor_make(w_shape, 4);
      tensor_t dw2 = tensor_make_alike(w2);
      char w2_name[MAX_STR_LENGTH];
      snprintf(w2_name, MAX_STR_LENGTH, "%s.conv2.weight", prefix);
      net_attach_param(model->list_all_params, w2_name, w2, dw2);

      // out
      uint out_shape[] = {N, 16, 3, 3};
      out = tensor_make(out_shape, 4);
      dout = tensor_make(out_shape, 4);
      snprintf(out_name, MAX_STR_LENGTH, "%s.out", prefix);
      net_attach_param(model->list_layer_out, out_name, out, dout);

      // cache
      snprintf(cache_name, MAX_STR_LENGTH, "%s.cache",prefix);
      net_attach_cache(model->list_layer_cache, cache_name);
    }
  }

  /* Pool */
  // input
  in = out;
  din = dout;

  snprintf(in_name, MAX_STR_LENGTH, "pool.in");
  net_attach_param(model->list_layer_in, in_name, in, din);

  // out
  uint pool_shape[] = {N, 16, 1, 1};
  out = tensor_make(pool_shape, 4);
  dout = tensor_make(pool_shape, 4);
  snprintf(out_name, MAX_STR_LENGTH, "pool.out");
  net_attach_param(model->list_layer_out, out_name, out, dout);

  // cache
  snprintf(cache_name, MAX_STR_LENGTH, "pool.cache");
  net_attach_cache(model->list_layer_cache, cache_name);

  /* FC */
  // in
  in = out;
  din = dout;
  snprintf(in_name, MAX_STR_LENGTH, "fc.in");
  net_attach_param(model->list_layer_in, in_name, in, din);

  // weight
  uint weight_shape_fc[] = {C, output_dim};
  tensor_t w_fc = tensor_make(weight_shape_fc, 2);
  tensor_t dw_fc = tensor_make_alike(w_fc);
  net_attach_param(model->list_all_params, "fc.weight", w_fc, dw_fc);

  // out
  uint fc_shape[] = {N, output_dim};
  out = tensor_make(fc_shape, 2);
  dout = tensor_make(fc_shape, 2);
  snprintf(out_name, MAX_STR_LENGTH, "fc.out");
  net_attach_param(model->list_layer_out, out_name, out, dout);

  // cache
  snprintf(cache_name, MAX_STR_LENGTH, "fc.cache");
  net_attach_cache(model->list_layer_cache, cache_name);

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
