/*
 * Resnet
 */
#include "awnn/layer_pool.h"
#include "awnn/layer_sandwich.h"
#include "awnn/loss_softmax.h"
#include "awnn/net_resnet.h"
#include "pthread.h"
#include "utils/data_cifar.h"
#include "utils/debug.h"
#include "utils/weight_init.h" 
#include "awnn/solver.h"

// #define ENABLE_SOLVER

static conv_param_t conv3x3_param = {.stride = 1, .padding = 1};
static conv_param_t conv3x3_with_sample_param = {.stride = 2, .padding = 1};

model_t * root_model = NULL;

status_t resnet_init(
    model_t *model,   // output
    dim_t input_dim,  // NCHW
    uint output_dim,  // nr_classes
    uint nr_stages,
    uint nr_blocks[MAX_STAGES],  // how many residual blocks in each stage
    T reg, normalize_method_t normalize_method) {
  init_helper_env();
  AWNN_NO_USE(normalize_method);
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

  uint filter_sz = 3;
  uint nr_filter = 16;

  /*
   * I.  preparation
   */
  // layer in
  in = tensor_make_placeholder(input_dim.dims, 4);
  din = tensor_make(input_dim.dims, 4);
  snprintf(in_name, MAX_STR_LENGTH, "conv1.in");
  net_attach_param(model->list_layer_in, in_name, in, din);

  // prepare and init weights
  uint w_shape[] = {nr_filter, C, filter_sz, filter_sz};
  tensor_t w = tensor_make(w_shape, 4);
  tensor_t dw = tensor_make(w_shape, 4);
  char w_name[MAX_STR_LENGTH];
  snprintf(w_name, MAX_STR_LENGTH, "conv1.weight");
  net_attach_param(model->list_all_params, w_name, w, dw);
  weight_init_kaiming(w);

  // layer out
  uint feature_sz =
      1 + (H + 2 * conv3x3_param.padding - filter_sz) / conv3x3_param.stride;
  AWNN_CHECK_EQ(feature_sz, 32);
  uint out_shape[] = {N, nr_filter, feature_sz, feature_sz};
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
    for (uint i_blk = 1; i_blk <= nr_blocks[i_stage - 1]; i_blk++) {
      char prefix[MAX_STR_LENGTH];
      snprintf(prefix, MAX_STR_LENGTH, "layer%u.%u", i_stage, i_blk);

      conv_param_t conv_param1, conv_param2;
      uint feature_sz;

      // In
      in = out;
      din = dout;
      snprintf(in_name, MAX_STR_LENGTH, "%s.in", prefix);
      net_attach_param(model->list_layer_in, in_name, in, din);

      if (i_stage > 1 &&
          i_blk == 1) {  // subsampling for for both identity and first conv
        uint w_sample_shape[] = {nr_filter, C, 1, 1};
        tensor_t w_sample = tensor_make(w_sample_shape, 4);
        tensor_t dw_sample = tensor_make_alike(w_sample);
        char w_sample_name[MAX_STR_LENGTH];
        snprintf(w_sample_name, MAX_STR_LENGTH, "%s.sample.weight", prefix);
        net_attach_param(model->list_all_params, w_sample_name, w_sample,
                         dw_sample);
        weight_init_kaiming(w_sample);

        conv_param1.padding = 1, conv_param1.stride = 2;
        feature_sz = in.dim.dims[3] / 2;
      } else {
        conv_param1.padding = 1, conv_param1.stride = 1;
        ;
        feature_sz = in.dim.dims[3];
      }
      conv_param2.padding = 1, conv_param2.stride = 1;

      // weight
      uint w_shape[] = {nr_filter, C, filter_sz, filter_sz};

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

      weight_init_kaiming(w1);
      weight_init_kaiming(w2);

      // out
      uint out_shape[] = {N, nr_filter, feature_sz, feature_sz};
      out = tensor_make(out_shape, 4);
      dout = tensor_make(out_shape, 4);
      snprintf(out_name, MAX_STR_LENGTH, "%s.out", prefix);
      net_attach_param(model->list_layer_out, out_name, out, dout);

      // cache
      snprintf(cache_name, MAX_STR_LENGTH, "%s.cache", prefix);
      net_attach_cache(model->list_layer_cache, cache_name);
    }
    C = nr_filter;
    nr_filter *= 2;
  }

  /* Pool */
  // input
  in = out;
  din = dout;

  snprintf(in_name, MAX_STR_LENGTH, "pool.in");
  net_attach_param(model->list_layer_in, in_name, in, din);

  // out
  uint pool_shape[] = {N, C, 1, 1};
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

  // bias
  uint bias_shape_fc[] = {output_dim};
  tensor_t b_fc = tensor_make(bias_shape_fc, 1);
  tensor_t db_fc = tensor_make_alike(b_fc);
  net_attach_param(model->list_all_params, "fc.bias", b_fc, db_fc);

  weight_init_fc_kaiming(w_fc, b_fc);

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
tensor_t _do_resnet_forward(model_t const *model, tensor_t x,
                            awnn_mode_t mode) {
  tensor_t layer_in = x;
  tensor_t layer_out;
  tensor_t w = net_get_param(model->list_all_params, "conv1.weight")->data;
  layer_out = net_get_param(model->list_layer_out, "conv1.out")->data;

  lcache_t *cache;
  if (mode == MODE_TRAIN)
    cache = net_get_cache(model->list_layer_cache, "conv1.cache");
  else
    cache = NULL;
  conv_relu_forward(layer_in, w, cache, conv3x3_param, layer_out);
  layer_in = layer_out;

  /*
   * II.  main stage
   */
  for (uint i_stage = 1; i_stage <= model->nr_stages; i_stage++) {
    for (uint i_blk = 1; i_blk <= model->nr_blocks[i_stage - 1]; i_blk++) {
      char prefix[MAX_STR_LENGTH];
      snprintf(prefix, MAX_STR_LENGTH, "layer%u.%u", i_stage, i_blk);

      char w1_name[MAX_STR_LENGTH], w2_name[MAX_STR_LENGTH];
      snprintf(w1_name, MAX_STR_LENGTH, "%s.conv1.weight", prefix);
      snprintf(w2_name, MAX_STR_LENGTH, "%s.conv2.weight", prefix);
      tensor_t w1 = net_get_param(model->list_all_params, w1_name)->data;
      tensor_t w2 = net_get_param(model->list_all_params, w2_name)->data;

      // locate preallocated layer_out
      char out_name[MAX_STR_LENGTH];
      snprintf(out_name, MAX_STR_LENGTH, "%s.out", prefix);
      layer_out = net_get_param(model->list_layer_out, out_name)->data;

      char cache_name[MAX_STR_LENGTH];
      snprintf(cache_name, MAX_STR_LENGTH, "%s.cache", prefix);
      if (mode == MODE_TRAIN)
        cache = net_get_cache(model->list_layer_cache, cache_name);
      else
        cache = NULL;

      if (i_stage > 1 &&
          i_blk == 1) {  // subsampling for for both identity and first conv
        char w_sample_name[MAX_STR_LENGTH];
        snprintf(w_sample_name, MAX_STR_LENGTH, "%s.sample.weight", prefix);
        tensor_t w_sample =
            net_get_param(model->list_all_params, w_sample_name)->data;
        residual_basic_no_bn_subspl_forward(layer_in, w_sample, w1, w2, cache,
                                            conv3x3_with_sample_param,
                                            conv3x3_param, layer_out);
      } else {
        residual_basic_no_bn_forward(layer_in, w1, w2, cache, conv3x3_param,
                                     layer_out);
      }
      layer_in = layer_out;
    }
  }

  /* Pool */
  layer_out = net_get_param(model->list_layer_out, "pool.out")->data;

  if (mode == MODE_TRAIN)
    cache = net_get_cache(model->list_layer_cache, "pool.cache");
  else
    cache = NULL;
  global_avg_pool_forward(layer_in, cache, layer_out);
  layer_in = layer_out;

  /* FC */
  tensor_t w_fc = net_get_param(model->list_all_params, "fc.weight")->data;
  tensor_t b_fc = net_get_param(model->list_all_params, "fc.bias")->data;
  layer_out = net_get_param(model->list_layer_out, "fc.out")->data;
  if (mode == MODE_TRAIN)
    cache = net_get_cache(model->list_layer_cache, "fc.cache");
  else
    cache = NULL;
  layer_fc_forward(layer_in, w_fc, b_fc, cache, layer_out);

  return layer_out;
}

tensor_t resnet_forward_infer(model_t const *model, tensor_t x) {
  return _do_resnet_forward(model, x, MODE_INFER);
}

tensor_t resnet_forward(model_t const *model, tensor_t x) {
  return _do_resnet_forward(model, x, MODE_TRAIN);
}

status_t resnet_backward(model_t const *model, tensor_t dout, T *ptr_loss) {
  T loss = 0;
  tensor_t din;
  lcache_t *cache;

  /* FC */
  din = net_get_param(model->list_layer_in, "fc.in")->diff;
  cache = net_get_cache(model->list_layer_cache, "fc.cache");
  tensor_t dw_fc = net_get_param(model->list_all_params, "fc.weight")->diff;
  tensor_t w_fc = net_get_param(model->list_all_params, "fc.weight")->data;
  tensor_t db_fc = net_get_param(model->list_all_params, "fc.bias")->diff;
  loss += 0.5 * (model->reg) * tensor_sum_of_square(w_fc);

  layer_fc_backward(din, dw_fc, db_fc, cache, dout);
  update_regulizer_gradient(w_fc, dw_fc, model->reg);
  dout = din;

  /* Pool*/
  din = net_get_param(model->list_layer_in, "pool.in")->diff;
  cache = net_get_cache(model->list_layer_cache, "pool.cache");
  assert(cache != NULL);

  global_avg_pool_backward(din, cache, dout);
  dout = din;

  /* residual blocks*/
  for (uint i_stage = model->nr_stages; i_stage != 0; i_stage--) {
    for (uint i_blk = model->nr_blocks[i_stage - 1]; i_blk != 0; i_blk--) {
      char prefix[MAX_STR_LENGTH];
      snprintf(prefix, MAX_STR_LENGTH, "layer%u.%u", i_stage, i_blk);

      // locate preallocated in
      char in_name[MAX_STR_LENGTH];
      snprintf(in_name, MAX_STR_LENGTH, "%s.in", prefix);
      din = net_get_param(model->list_layer_in, in_name)->diff;

      char w1_name[MAX_STR_LENGTH], w2_name[MAX_STR_LENGTH];
      snprintf(w1_name, MAX_STR_LENGTH, "%s.conv1.weight", prefix);
      snprintf(w2_name, MAX_STR_LENGTH, "%s.conv2.weight", prefix);
      tensor_t dw1 = net_get_param(model->list_all_params, w1_name)->diff;
      tensor_t dw2 = net_get_param(model->list_all_params, w2_name)->diff;
      tensor_t w1 = net_get_param(model->list_all_params, w1_name)->data;
      tensor_t w2 = net_get_param(model->list_all_params, w2_name)->data;
      loss += 0.5 * (model->reg) * tensor_sum_of_square(w1);
      loss += 0.5 * (model->reg) * tensor_sum_of_square(w2);

      char cache_name[MAX_STR_LENGTH];
      snprintf(cache_name, MAX_STR_LENGTH, "%s.cache", prefix);
      cache = net_get_cache(model->list_layer_cache, cache_name);

      if (i_stage > 1 &&
          i_blk == 1) {  // subsampling for for both identity and first conv
        char w_sample_name[MAX_STR_LENGTH];
        snprintf(w_sample_name, MAX_STR_LENGTH, "%s.sample.weight", prefix);
        tensor_t w_sample =
            net_get_param(model->list_all_params, w_sample_name)->data;
        tensor_t dw_sample =
            net_get_param(model->list_all_params, w_sample_name)->diff;
        loss += 0.5 * (model->reg) * tensor_sum_of_square(w_sample);
        residual_basic_no_bn_subspl_backward(din, dw_sample, dw1, dw2, cache,
                                             conv3x3_with_sample_param,
                                             conv3x3_param, dout);
      } else {
        residual_basic_no_bn_backward(din, dw1, dw2, cache, conv3x3_param,
                                      dout);
      }

      update_regulizer_gradient(w1, dw1, model->reg);
      update_regulizer_gradient(w2, dw2, model->reg);
      dout = din;
    }
  }
  /* Preparation stage */
  din = net_get_param(model->list_layer_in, "conv1.in")->diff;
  tensor_t dw = net_get_param(model->list_all_params, "conv1.weight")->diff;
  tensor_t w = net_get_param(model->list_all_params, "conv1.weight")->data;
  loss += 0.5 * (model->reg) * tensor_sum_of_square(w);

  cache = net_get_cache(model->list_layer_cache, "conv1.cache");
  conv_relu_backward(din, dw, cache, conv3x3_param, dout);
  // TODO: regulizer
  update_regulizer_gradient(w, dw, model->reg);
  *ptr_loss = loss;
  return S_OK;
}

/**
 * Compute loss for a batch of (x,y), do forward/backward, and update
 * gradients*/
status_t resnet_loss(model_t const *model, tensor_t x, label_t const labels[],
                     T *ptr_loss) {
  T loss_classify, loss_reg;
  PDBG("========= Forwarding ==========");
  tensor_t out, dout;

  // Forward
  resnet_forward(model, x);

  // Softmax
  param_t *param_score = net_get_param(model->list_layer_out, "fc.out");
  AWNN_CHECK_NE(NULL, labels);
  out = param_score->data;
  dout = param_score->diff;
  PDBG("========= Softmax ==========");
  AWNN_CHECK_EQ(S_OK,
                loss_softmax(out, labels, &loss_classify, MODE_TRAIN, dout));

  // Backward
  PDBG("========= Backwarding ==========");
  AWNN_CHECK_EQ(S_OK, resnet_backward(model, dout, &loss_reg));

  *ptr_loss = loss_classify + loss_reg;
  return S_OK;
}

/** Naive all-reduce between all threads*/
void concurrent_allreduce_gradient(resnet_thread_info_t *worker_info){

  // Accumulate all gradients from worker to main
  pthread_barrier_wait(worker_info->ptr_barrier);
  if(worker_info->id != 0){
    // all learnable parameters 
    param_t *ptr_param_root;
    param_t *ptr_param_local;
    // similar to what i have done with solver
    list_for_each_entry_pairwise(ptr_param_root, root_model->list_all_params,
        ptr_param_local, worker_info->model.list_all_params, list) {
      PDBG("Accumulate %s...", ptr_param_local->name);
      //TODO: can be more fine grain
      
      pthread_mutex_lock(worker_info->ptr_mutex);
      tensor_elemwise_op_inplace(ptr_param_root->diff, ptr_param_local->diff, TENSOR_OP_ADD);
      pthread_mutex_unlock(worker_info->ptr_mutex);
    }
  }

  // main thread get avg
  pthread_barrier_wait(worker_info->ptr_barrier);
  if(worker_info->id == 0 && worker_info->nr_threads > 1){
    param_t *ptr_param_root;
    // similar to what i have done with solver
    list_for_each_entry(ptr_param_root, root_model->list_all_params, list) {
      PDBG("Averaging gradient in root %s...", ptr_param_root->name);
      T *pelem;
      uint ii;
      tensor_t dparam = ptr_param_root->diff;
      tensor_for_each_entry(pelem, ii, dparam) { (*pelem) /= (worker_info->nr_threads); }
    }
  }

  // each other thread get a copy
  pthread_barrier_wait(worker_info->ptr_barrier);
  if(worker_info->id != 0){
    param_t *ptr_param_root;
    param_t *ptr_param_local;
    // similar to what i have done with solver
    list_for_each_entry_pairwise(ptr_param_root, root_model->list_all_params,
        ptr_param_local, worker_info->model.list_all_params, list) {
      PDBG("Obatining gradient copy %s...", ptr_param_local->name);
      tensor_copy(ptr_param_local->diff, ptr_param_root->diff);
    }
  }

  pthread_barrier_wait(worker_info->ptr_barrier);
}


void *resnet_thread_entry(void *threadinfo) {
  struct resnet_thread_info *my_info =
      (struct resnet_thread_info *)(threadinfo);
  tensor_t x_thread_local;
  label_t *labels_thread_local;

  /* Split batch data to all thread*/
  uint cur_batch = 0;
  uint cnt_read = get_train_batch_mt(
      my_info->data_loader, &x_thread_local, &labels_thread_local, cur_batch,
      my_info->batch_sz, my_info->id, my_info->nr_threads);

  AWNN_CHECK_EQ(my_info->batch_sz, cnt_read * my_info->nr_threads);

  /* Network config*/
  dim_t input_dim = {.dims = {cnt_read, 3, 32, 32}};
  uint output_dim = 10;
  uint nr_stages = 1;
  uint nr_blocks[MAX_STAGES] = {2};
  T reg = 1;
  normalize_method_t normalize_method = NORMALIZE_NONE;  // no batchnorm now

  set_conv_method(CONV_METHOD_PERIMG);

  /* Allocate spaces for input/output/layer cache/weight/gradient*/
  resnet_init(&(my_info->model), input_dim, output_dim, nr_stages, nr_blocks,
              reg, normalize_method);

  if (my_info->id == 0) {
    root_model = &(my_info->model);
  };

  AWNN_CHECK_EQ((void *)0, (void *)net_get_param(my_info->model.list_all_params,
                                                 "W3"));  // unexisting param
  AWNN_CHECK_NE((void *)0, (void *)net_get_param(my_info->model.list_all_params,
                                                 "conv1.weight"));

  T loss = 0;
  /*
  resnet_loss(&(my_info->model), x_thread_local, labels_thread_local, &loss);
  PINF("Initial Loss %.2f", loss);
  PINF("Using convolution method %d", get_conv_method());
  */
  uint nr_iterations = 10;
  clocktime_t t_start;
  double forward_backward_in_ms = 0;
  double allreduce_in_ms = 0;
  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    if (my_info->id == 0) {
      t_start = get_clocktime();
    };
    resnet_loss(&(my_info->model), x_thread_local, labels_thread_local, &loss);
    if (my_info->id == 0) {
      PINF("worker%d, Iter=%u, Loss %.2f", my_info->id, iteration, loss);
      forward_backward_in_ms += get_elapsed_ms(t_start, get_clocktime());
    };

    if (my_info->id == 0) {
      t_start = get_clocktime();
    };
    concurrent_allreduce_gradient(my_info);
#ifdef ENABLE_SOLVER
    param_t *p_param;
    // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
    list_for_each_entry(p_param, my_info->model.list_all_params, list) {
      PDBG("updating %s...", p_param->name);
      // sgd
      // sgd_update(p_param, learning_rate);
      sgd_update_momentum(p_param, 0.01, 0.9);
    }
#endif

    if (my_info->id == 0) {
      allreduce_in_ms += get_elapsed_ms(t_start, get_clocktime());
    };
  }
  if (my_info->id == 0) {
    PWRN("AVG forward-backward %.3fms, allreduce(%.3f)ms", forward_backward_in_ms/nr_iterations, allreduce_in_ms);
  }

  resnet_finalize(&(my_info->model));
  pthread_exit((void *)threadinfo);
  return NULL;
}
