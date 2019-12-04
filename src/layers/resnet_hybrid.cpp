#include "awnn/tensor.h"
#include "layers/layer_common.hpp"
#include "utils/debug.h"
#include "utils/weight_init.h"
#include "awnn/memory.h"
#define ENABLE_SOLVER

// TODO: i had use global varaibles otherwise dimension info will be lost

extern layer_conv2d_config_t conv_config;
extern layer_resblock_config_t resblock_config;
extern layer_pool_config_t pool_config;
extern layer_fc_config_t fc_config;

void resnet_setup_hybrid(net_t *net, uint input_shape[], double reg, topo_config_t *topo) {
  paral_config_t paral_config = PARAL_TYPE_DATA; // all layers using data parallelism
  /*Conv layer*/
  net->dataconfig.name = "data";

  net->dataconfig.dim.dims[0] = input_shape[0];
  net->dataconfig.dim.dims[1] = input_shape[1];
  net->dataconfig.dim.dims[2] = input_shape[2];
  net->dataconfig.dim.dims[3] = input_shape[3];

  layer_t *data_layer =
      layer_setup(LAYER_TYPE_DATA, &(net->dataconfig), nullptr, topo, paral_config);
  net_add_layer(net, data_layer);

  /*Conv layer*/
  conv_config.name = "conv2d";
  conv_config.out_channels = 16;
  conv_config.kernel_size = 3;
  conv_config.reg = reg;
  conv_config.activation = ACTIVATION_RELU;

  layer_t *conv_layer =
      layer_setup(LAYER_TYPE_CONV2D, &conv_config, data_layer, topo, paral_config);
  net_add_layer(net, conv_layer);

  /*First residual block*/
  resblock_config.name = "resblock";
  resblock_config.reg = reg;

  layer_t *resblock_layer =
      layer_setup(LAYER_TYPE_RESBLOCK, &resblock_config, conv_layer, topo, paral_config);
  net_add_layer(net, resblock_layer);

  /*pool layer*/
  pool_config.name = "pool";
  pool_config.type = POOL_GLOBAL_AVG;

  layer_t *pool_layer =
      layer_setup(LAYER_TYPE_POOL, &pool_config, resblock_layer, topo, paral_config);
  net_add_layer(net, pool_layer);

  /*FC layer*/
  fc_config.name = "fc";
  fc_config.nr_classes = 10;
  fc_config.reg = reg;
  fc_config.activation = ACTIVATION_NONE;

  layer_t *fc_layer = layer_setup(LAYER_TYPE_FC, &fc_config, pool_layer, topo, paral_config);
  net_add_layer(net, fc_layer);
}

void *resnet_main(int batch_size, int nr_threads, int nr_iterations) {

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

  resnet_setup_hybrid(&(model), in_shape, reg, &topology);

  /* Data loader*/
  data_loader_t loader;
  status_t ret = cifar_open_batched(&loader, CIFAR_PATH, batch_size, nr_threads);
  uint train_sz = 4000;
  uint val_sz = 1000;
  AWNN_CHECK_EQ(S_OK, ret);
  AWNN_CHECK_EQ(S_OK, cifar_split_train(&loader, train_sz, val_sz));

  struct concurrent_context *context = (struct concurrent_context *)mem_zalloc(sizeof(struct concurrent_context));
  context->loader = &loader;
  context->net = &model;
  context->reg_losses = (double*) mem_zalloc(sizeof(double)*nr_threads);
  context->classify_losses = (double*) mem_zalloc(sizeof(double)*nr_threads);
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
  resnet_loss(&(my_info->model), x_thread_local, labels_thread_local, &loss);
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

  resnet_teardown(&model);

  cifar_close(&loader);

  mem_free(context->reg_losses);
  mem_free(context->classify_losses);
  mem_free(context);

  return NULL;
}

