#include "awnn/tensor.h"
#include "layers/layer_common.hpp"
#include "utils/debug.h"
#include "utils/weight_init.h"
#define ENABLE_SOLVER

// TODO: i had use global varaibles otherwise dimension info will be lost

net_t *root_model = NULL;
layer_conv2d_config_t conv_config;
layer_resblock_config_t resblock_config;
layer_pool_config_t pool_config;
layer_fc_config_t fc_config;

void resnet_setup(net_t *net, uint input_shape[], double reg) {
  /*Conv layer*/
  net->dataconfig.name = "data";

  net->dataconfig.dim.dims[0] = input_shape[0];
  net->dataconfig.dim.dims[1] = input_shape[1];
  net->dataconfig.dim.dims[2] = input_shape[2];
  net->dataconfig.dim.dims[3] = input_shape[3];

  layer_t *data_layer =
      layer_setup(LAYER_TYPE_DATA, &(net->dataconfig), nullptr);
  net_add_layer(net, data_layer);

  /*Conv layer*/
  conv_config.name = "conv2d";
  conv_config.out_channels = 16;
  conv_config.kernel_size = 3;
  conv_config.reg = reg;
  conv_config.activation = ACTIVATION_RELU;

  layer_t *conv_layer =
      layer_setup(LAYER_TYPE_CONV2D, &conv_config, data_layer);
  net_add_layer(net, conv_layer);

  /*First residual block*/
  resblock_config.name = "resblock";
  resblock_config.reg = reg;

  layer_t *resblock_layer =
      layer_setup(LAYER_TYPE_RESBLOCK, &resblock_config, conv_layer);
  net_add_layer(net, resblock_layer);

  /*pool layer*/
  pool_config.name = "pool";

  layer_t *pool_layer =
      layer_setup(LAYER_TYPE_POOL, &pool_config, resblock_layer);
  net_add_layer(net, pool_layer);

  /*FC layer*/
  fc_config.name = "fc";
  fc_config.nr_classes = 10;
  fc_config.reg = reg;
  fc_config.activation = ACTIVATION_NONE;

  layer_t *fc_layer = layer_setup(LAYER_TYPE_FC, &fc_config, pool_layer);
  net_add_layer(net, fc_layer);
}

void resnet_teardown(net_t *net) { net_teardown(net); }

/** Naive all-reduce between all threads*/
void concurrent_allreduce_gradient(resnet_thread_info_t *worker_info) {
  // Accumulate all gradients from worker to main
  pthread_barrier_wait(worker_info->ptr_barrier);
  net_t *local_model = &(worker_info->model);
  net_t *global_model = root_model;
  if (worker_info->id != 0) {
    for (size_t idx_layer = 0; idx_layer < local_model->layers.size();
         idx_layer++) {
      size_t nr_learnables_this_layer =
          local_model->layers[idx_layer]->learnables.size();
      for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
           idx_param++) {
        Blob *param_local =
            local_model->layers[idx_layer]->learnables[idx_param];
        Blob *param_global =
            global_model->layers[idx_layer]->learnables[idx_param];
        PDBG("updating %s...", param_local->name.c_str());
        AWNN_CHECK_EQ(param_local->learnable, 1);
        // sgd
        // sgd_update(p_param, learning_rate);
        pthread_mutex_lock(worker_info->ptr_mutex);
        tensor_elemwise_op_inplace((param_global)->diff[0], (param_local)->diff[0],
                                   TENSOR_OP_ADD);
        pthread_mutex_unlock(worker_info->ptr_mutex);
      }
    }
  }

  // main thread get avg
  pthread_barrier_wait(worker_info->ptr_barrier);
  if (worker_info->id == 0 && worker_info->nr_threads > 1) {
    for (size_t idx_layer = 0; idx_layer < local_model->layers.size();
         idx_layer++) {
      size_t nr_learnables_this_layer =
          local_model->layers[idx_layer]->learnables.size();
      for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
           idx_param++) {
        Blob *param_local =
            local_model->layers[idx_layer]->learnables[idx_param];
        PDBG("averaging %s...", (param_local)->name.c_str());

        uint ii;
        tensor_t dparam = (param_local)->diff[0];

        uint capacity = tensor_get_capacity(dparam);
        for(ii = 0;  ii < capacity; ii++){
          dparam.data[ii] /= (worker_info->nr_threads);
        }
      }
    }
  }

  // each other thread get a copy
  pthread_barrier_wait(worker_info->ptr_barrier);
  if (worker_info->id != 0) {
    for (size_t idx_layer = 0; idx_layer < local_model->layers.size();
         idx_layer++) {
      size_t nr_learnables_this_layer =
          local_model->layers[idx_layer]->learnables.size();
      for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
           idx_param++) {
        Blob *param_local =
            local_model->layers[idx_layer]->learnables[idx_param];
        Blob *param_global =
            global_model->layers[idx_layer]->learnables[idx_param];

        PDBG("Duplicating %s...", (param_local)->name.c_str());
        AWNN_CHECK_EQ((param_local)->learnable, 1);
        tensor_copy((param_local)->diff[0], (param_global)->diff[0]);
      }
    }
  }

  pthread_barrier_wait(worker_info->ptr_barrier);
}

void *resnet_thread_entry(void *threadinfo) {
  struct resnet_thread_info *my_info =
      (struct resnet_thread_info *)(threadinfo);
  struct reader_local_info * reader_info = my_info->data_loader->readers_info + my_info->id;
  tensor_t *x_thread_local  = &(reader_info->cur_x);
  label_t **labels_thread_local = &(reader_info->cur_label);

  /* Split batch data to all thread*/
  uint cur_batch = 0;
  uint cnt_read = get_train_batch_mt(
      my_info->data_loader, (uint)my_info->id);

  /* Network config*/
  uint in_shape[] = {cnt_read, 3, 32, 32};
  PINF("thread[%d], split sizse: %u", my_info->id, cnt_read);
  T reg = 0.001;

  resnet_setup(&(my_info->model), in_shape, reg);

  if (my_info->id == 0) {
    root_model = &(my_info->model);
  };

  T loss = 0;
  /*
  resnet_loss(&(my_info->model), x_thread_local, labels_thread_local, &loss);
  PINF("Using convolution method %d", get_conv_method());
  */
  uint nr_iterations = my_info->nr_iterations;
  clocktime_t t_start;
  double forward_backward_in_ms = 0;
  double allreduce_in_ms = 0;
  double gradientupdate_in_ms = 0;
  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    if (my_info->id == 0) {
      t_start = get_clocktime();
    };

    /** Forward/backward*/
    net_loss(&(my_info->model), *x_thread_local, *labels_thread_local, &loss, 0);
    if (my_info->id == 0) {
      PINF("worker%d, Iter=%u, Loss %.2f", my_info->id, iteration, loss);
      forward_backward_in_ms += get_elapsed_ms(t_start, get_clocktime());
    };

    /** All reduce gradient*/
    if (my_info->id == 0) {
      t_start = get_clocktime();
    };
    concurrent_allreduce_gradient(my_info);
    if (my_info->id == 0) {
      allreduce_in_ms += get_elapsed_ms(t_start, get_clocktime());
    };

      /** Update gradient*/
#ifdef ENABLE_SOLVER
    if (my_info->id == 0) {
      t_start = get_clocktime();
    };

    net_update_weights(&(my_info->model), 0.01);

    if (my_info->id == 0) {
      gradientupdate_in_ms += get_elapsed_ms(t_start, get_clocktime());
    };

#endif
  }
  if (my_info->id == 0) {
    double time_per_iter = forward_backward_in_ms / nr_iterations +
                           allreduce_in_ms / nr_iterations +
                           gradientupdate_in_ms / nr_iterations;
    PWRN(
        "[thread 0]: time-per-iteration (%.3f ms), forward-backward (%.3f ms), "
        "allreduce (%.3f ms), gradientupdate (%.3f ms)",
        time_per_iter, forward_backward_in_ms / nr_iterations,
        allreduce_in_ms / nr_iterations, gradientupdate_in_ms / nr_iterations);
  }

  resnet_teardown(&(my_info->model));
  pthread_exit((void *)threadinfo);
  return NULL;
}
