// see issue #61

#include "awnn/memory.h"
#include "awnn/tensor.h"
#include "layers/layer_common.hpp"
#include "utils/debug.h"
#include "utils/weight_init.h"
#define ENABLE_SOLVER

// TODO: i had use global varaibles otherwise dimension info will be lost

extern layer_conv2d_config_t conv_config;
extern layer_resblock_config_t resblock_config;
extern layer_pool_config_t pool_config;
layer_fc_config_t fc1_config;
layer_fc_config_t fc2_config;
layer_fc_config_t fc3_config;

void vggnet_teardown(net_t *net) { net_teardown(net); }

void vggnet_setup_hybrid(net_t *net, uint input_shape[], double reg,
                         topo_config_t *topo) {
  paral_config_t paral_config =
      PARAL_TYPE_DATA;  // all layers using data parallelism
  /*Conv layer*/
  net->dataconfig.name = "data";

  net->dataconfig.dim.dims[0] = input_shape[0];
  net->dataconfig.dim.dims[1] = input_shape[1];
  net->dataconfig.dim.dims[2] = input_shape[2];
  net->dataconfig.dim.dims[3] = input_shape[3];

  layer_t *data_layer = layer_setup(LAYER_TYPE_DATA, &(net->dataconfig),
                                    nullptr, topo, paral_config);
  net_add_layer(net, data_layer);

  /*Conv layer->1, 64, 32, 32*/
  conv_config.name = "conv2d";
  conv_config.out_channels = 64;
  conv_config.kernel_size = 3;
  conv_config.reg = reg;
  conv_config.activation = ACTIVATION_RELU;

  layer_t *conv_layer = layer_setup(LAYER_TYPE_CONV2D, &conv_config, data_layer,
                                    topo, paral_config);
  net_add_layer(net, conv_layer);

  pool_config.name = "pool";
  pool_config.type = POOL_MAX;
  pool_config.kernel_size = 2;

  layer_t *pool_layer = layer_setup(LAYER_TYPE_POOL, &pool_config, conv_layer,
                                    topo, paral_config);
  net_add_layer(net, pool_layer);

  /*FC layer*/
  fc1_config.name = "fc1";
  fc1_config.nr_classes = 1024;
  fc1_config.reg = reg;
  fc1_config.activation = ACTIVATION_RELU;

  layer_t *fc1_layer =
      layer_setup(LAYER_TYPE_FC, &fc1_config, pool_layer, topo, paral_config);
  net_add_layer(net, fc1_layer);

  /*FC layer*/
  fc2_config.name = "fc2";
  fc2_config.nr_classes = 256;
  fc2_config.reg = reg;
  fc2_config.activation = ACTIVATION_RELU;

  layer_t *fc2_layer =
      layer_setup(LAYER_TYPE_FC, &fc2_config, fc1_layer, topo, paral_config);
  net_add_layer(net, fc2_layer);

  /*FC layer*/
  fc3_config.name = "fc3";
  fc3_config.nr_classes = 10;
  fc3_config.reg = reg;
  fc3_config.activation = ACTIVATION_NONE;

  layer_t *fc3_layer =
      layer_setup(LAYER_TYPE_FC, &fc3_config, fc2_layer, topo, paral_config);
  net_add_layer(net, fc3_layer);
}

void *vggnet_main(int batch_size, int nr_threads, int nr_iterations) {
  net_t model;

  topo_config_t topology(nr_threads);
#if 0
  /* Split batch data to all thread TODO: split reading to multiple instances*/
  uint cur_batch = 0;
  uint cnt_read = get_train_batch_mt(
      my_info->data_loader, &x_thread_local, &labels_thread_local, cur_batch,
      my_info->batch_sz, (uint)my_info->id, (uint)my_info->nr_threads);
#endif

  /* Network config*/
  uint in_shape[] = {(uint)batch_size, 3, 32, 32};
  T reg = 0.001;

  vggnet_setup_hybrid(&(model), in_shape, reg, &topology);

  /* Data loader*/
  data_loader_t loader;
  status_t ret =
      cifar_open_batched(&loader, CIFAR_PATH, batch_size, nr_threads);
  uint train_sz = 4000;
  uint val_sz = 1000;
  AWNN_CHECK_EQ(S_OK, ret);
  AWNN_CHECK_EQ(S_OK, cifar_split_train(&loader, train_sz, val_sz));

  struct concurrent_context *context = (struct concurrent_context *)mem_zalloc(
      sizeof(struct concurrent_context));
  context->loader = &loader;
  context->net = &model;
  context->reg_losses = (double *)mem_zalloc(sizeof(double) * nr_threads);
  context->classify_losses = (double *)mem_zalloc(sizeof(double) * nr_threads);
  context->topo = &topology;
  context->lr = 0.01;

  pthread_mutex_t mutex;
  pthread_mutex_init(&mutex, NULL);
  context->ptr_mutex = &mutex;

  /* Used for all-reduce*/
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, nr_threads);
  context->ptr_barrier = &barrier;

  double loss = 0;
  /*
  vggnet_loss(&(my_info->model), x_thread_local, labels_thread_local, &loss);
  PINF("Using convolution method %d", get_conv_method());
  */
  clocktime_t t_start;
  double forward_backward_in_ms = 0;
  double allreduce_in_ms = 0;
  double gradientupdate_in_ms = 0;
  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    t_start = get_clocktime();

    /** Forward/backward*/
    net_loss_hybrid(context, &loss, 0);

    PINF("Iter=%u, Loss(worker 1) %.2f", iteration, loss);
    forward_backward_in_ms += get_elapsed_ms(t_start, get_clocktime());

    /** All reduce gradient*/
    t_start = get_clocktime();

    // concurrent_allreduce_gradient(my_info);
    allreduce_hybrid(context);
    //
    allreduce_in_ms += get_elapsed_ms(t_start, get_clocktime());

    /** Update gradient*/
#ifdef ENABLE_SOLVER
    t_start = get_clocktime();

    net_update_weights_hybrid(context);

    gradientupdate_in_ms += get_elapsed_ms(t_start, get_clocktime());
#endif
  }

  double time_per_iter = forward_backward_in_ms / nr_iterations +
                         allreduce_in_ms / nr_iterations +
                         gradientupdate_in_ms / nr_iterations;
  PWRN(
      "[thread 0]: time-per-iteration (%.3f ms), forward-backward (%.3f ms), "
      "allreduce (%.3f ms), gradientupdate (%.3f ms)",
      time_per_iter, forward_backward_in_ms / nr_iterations,
      allreduce_in_ms / nr_iterations, gradientupdate_in_ms / nr_iterations);

  vggnet_teardown(&model);

  cifar_close(&loader);

  mem_free(context->reg_losses);
  mem_free(context->classify_losses);
  mem_free(context);

  return NULL;
}
