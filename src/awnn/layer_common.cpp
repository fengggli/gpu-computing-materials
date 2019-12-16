#include "layer_common.hpp"
#include "awnn/common.h"
#include "awnn/layer_conv.h"
#include "awnn/layer_fc.h"
#include "awnn/layer_pool.h"
#include "awnn/loss_softmax.h"
#include "awnn/net_resnet.h"
#include "awnn/solver.h"
#include "utils/weight_init.h"
#define DO_AVERAGE

#define IsPowerOf2(n) (((n)&(n-1)) == 0)
uint Power2RoundUp(uint v){
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

/** I only need those layers:
 * 1. conv, relu, and conv_relu
 * 2. last fc_layer,
 * 3. residual blocks*/

void layer_data_setup(layer_t *this_layer, layer_data_config_t *layer_config,
                      layer_t *bottom_layer) {
  this_layer->layer_type = LAYER_TYPE_DATA;
  AWNN_CHECK_EQ(bottom_layer, nullptr);
  this_layer->name = layer_config->name;
  this_layer->layer_in = nullptr;

  /*Calculate output shape*/
  if (this_layer->paral_config != PARAL_TYPE_DATA) {
    PERR("paral type %d not supported in FC layer", this_layer->paral_config);
    exit(-1);
  } else {
    this_layer->layer_out =
        new Blob(this_layer->name + ".out", 0, layer_config->dim.dims,
                 DATA_PARTITIONED_N, this_layer->topo);
    return;
  }
}

static inline void _do_inplace_relu_forward(tensor_t y) {
  uint capacity = tensor_get_capacity(y);
  for (uint i = 0; i < capacity; i++) {
    T *elem = y.data;
    elem[i] = elem[i] > 0 ? elem[i] : 0;
  }
}

static inline void _do_inplace_relu_backward(tensor_t dx, tensor_t x) {
  uint capacity = tensor_get_capacity(dx);
  for (uint i = 0; i < capacity; i++) {
    T *elem = dx.data;
    elem[i] = x.data[i] > 0 ? elem[i] : 0;
  }
}

// Inplace layer, y and x points to same tensor
double layer_relu_forward(tensor_t x, tape_t &tape, tensor_t y,
                          void *layer_config, int id) {
  _do_inplace_relu_forward(y);
  return 0;
}

void layer_relu_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                         void *layer_config, int id) {
  tensor_t x = tape["in"]->data[id];
  _do_inplace_relu_backward(dx, x);
}

void layer_relu_setup(layer_t *this_layer, layer_relu_config_t *layer_config,
                      layer_t *bottom_layer) {
  this_layer->forward = &layer_relu_forward;
  this_layer->backward = &layer_relu_backward;
  this_layer->layer_type = LAYER_TYPE_RELU_INPLACE;

  this_layer->name = layer_config->name;

  if (this_layer->paral_config != PARAL_TYPE_DATA) {
    PERR("paral type %d not supported in FC layer", this_layer->paral_config);
    exit(-1);
  } else {
    this_layer->layer_in = bottom_layer->layer_out;
    this_layer->layer_out = bottom_layer->layer_out;

    this_layer->tape.insert({"in", this_layer->layer_in});
  }
}

// global averge pool layer
double layer_pool_forward(tensor_t x, tape_t &tape, tensor_t y,
                          void *layer_config, int id) {
  layer_pool_config_t *config = (layer_pool_config_t *)(layer_config);

  if (config->type == POOL_GLOBAL_AVG) {
    do_global_pool_forward(x, y);
  } else {
    do_max_pool_forward(x, y, config->kernel_size);
  }
  return 0;
}

void layer_pool_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                         void *layer_config, int id) {
  layer_pool_config_t *config = (layer_pool_config_t *)(layer_config);

  tensor_t x = tape["in"]->data[id];
  tensor_t y = tape["out"]->data[id];

  if (config->type == POOL_GLOBAL_AVG) {
    do_global_pool_backward(dx, dy);
  } else {
    do_max_pool_backward(dx, dy, config->kernel_size, x, y);
  }
}

void layer_pool_setup(layer_t *this_layer, layer_pool_config_t *layer_config,
                      layer_t *bottom_layer) {
  this_layer->forward = &layer_pool_forward;
  this_layer->backward = &layer_pool_backward;
  this_layer->layer_type = LAYER_TYPE_POOL;

  this_layer->name = layer_config->name;

  if (this_layer->paral_config != PARAL_TYPE_DATA) {
    PERR("paral type %d not supported in FC layer", this_layer->paral_config);
    exit(-1);
  } else {
    this_layer->layer_in = bottom_layer->layer_out;

    uint nr_imgs = this_layer->layer_in->global_dim.dims[0];
    uint in_channels = this_layer->layer_in->global_dim.dims[1];
    uint in_height = this_layer->layer_in->global_dim.dims[2];

    /*Calculate output shape*/
    uint out_size;
    if (layer_config->type == POOL_GLOBAL_AVG)
      out_size = 1;
    else
      out_size = in_height / layer_config->kernel_size;

    uint out_shape[] = {nr_imgs, in_channels, out_size, out_size};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape,
                                     DATA_PARTITIONED_N, this_layer->topo);

    this_layer->tape.insert({"in", this_layer->layer_in});    // save x
    this_layer->tape.insert({"out", this_layer->layer_out});  // save x
  }
}

double layer_conv2d_forward(tensor_t x, tape_t &tape, tensor_t y,
                            void *layer_config, int id) {
  double reg_loss = 0;
  layer_conv2d_config_t *config = (layer_conv2d_config_t *)(layer_config);
  tensor_t w = tape["weight"]->data[id];
  do_conv_forward_perimg(x, w, y, config->padding, config->stride);

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_forward(y);
  }

  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(w);
  return reg_loss;
}

void layer_conv2d_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                           void *layer_config, int id) {
  layer_conv2d_config_t *config = (layer_conv2d_config_t *)(layer_config);
  tensor_t x = tape["in"]->data[id];
  tensor_t w = tape["weight"]->data[id];
  tensor_t dw = tape["weight"]->diff[id];
  tensor_t y = tape["out"]->data[id];

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_backward(dy, y);
  }

  do_conv_backward_perimg(dx, dw, dy, x, w, config->padding, config->stride);

  if (config->reg > 0) update_regulizer_gradient(w, dw, config->reg);
}

void layer_conv2d_setup(layer_t *this_layer,
                        layer_conv2d_config_t *layer_config,
                        layer_t *bottom_layer) {
  /* Setup Forward/backward*/
  this_layer->forward = &layer_conv2d_forward;
  this_layer->backward = &layer_conv2d_backward;
  AWNN_CHECK_GT(layer_config->out_channels, 0);
  AWNN_CHECK_GT(layer_config->kernel_size, 0);

  this_layer->name = layer_config->name;

  if (this_layer->paral_config != PARAL_TYPE_DATA) {
    PERR("paral type %d not supported in FC layer", this_layer->paral_config);
    exit(-1);
  } else {
    this_layer->layer_in = bottom_layer->layer_out;

    uint nr_imgs = this_layer->layer_in->global_dim.dims[0];
    uint in_channels = this_layer->layer_in->global_dim.dims[1];
    uint in_height = this_layer->layer_in->global_dim.dims[2];

    /* Allocate weight*/
    uint w_shape[] = {layer_config->out_channels, in_channels,
                      layer_config->kernel_size, layer_config->kernel_size};
    Blob *weight_blob = new Blob(this_layer->name + ".weight", 1, w_shape,
                                 DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight_blob);

    /* Weight init*/
    AWNN_CHECK_EQ(S_OK, weight_init_kaiming(weight_blob->data[0]));

    /*Calculate output shape*/
    uint out_height = 1 + (in_height + 2 * layer_config->padding -
                           layer_config->kernel_size) /
                              layer_config->stride;
    uint out_shape[] = {nr_imgs, layer_config->out_channels, out_height,
                        out_height};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape,
                                     DATA_PARTITIONED_N, this_layer->topo);

    /* Set tensor other than input/output*/
    this_layer->tape.insert({"in", this_layer->layer_in});    // save x
    this_layer->tape.insert({"weight", weight_blob});         // save w
    this_layer->tape.insert({"out", this_layer->layer_out});  // save y

    return;
  }
}

double layer_fc_forward(tensor_t x, tape_t &tape, tensor_t y,
                        void *layer_config, int id) {
  double reg_loss = 0;
  layer_fc_config_t *config = (layer_fc_config_t *)(layer_config);

  tensor_t w = tape["weight"]->data[id];
  tensor_t b = tape["bias"]->data[id];
  do_layer_fc_forward(x, w, b, y);

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_forward(y);
  }

  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(w);
  // PINF("regloss %.3f", reg_loss);
  return reg_loss;
}

void layer_fc_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                       void *layer_config, int id) {
  layer_fc_config_t *config = (layer_fc_config_t *)(layer_config);

  tensor_t dw = tape["weight"]->diff[id];
  tensor_t db = tape["bias"]->diff[id];
  tensor_t x = tape["in"]->data[id];
  tensor_t w = tape["weight"]->data[id];
  tensor_t y = tape["out"]->data[id];

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_backward(dy, y);
  }

  do_layer_fc_backward(dx, dw, db, dy, x, w);

  if (config->reg > 0) update_regulizer_gradient(w, dw, config->reg);
}

void layer_fc_setup(layer_t *this_layer, layer_fc_config_t *layer_config,
                    layer_t *bottom_layer) {
  this_layer->forward = &layer_fc_forward;
  this_layer->backward = &layer_fc_backward;
  AWNN_CHECK_GT(layer_config->nr_classes, 0);
  this_layer->layer_type = LAYER_TYPE_FC;

  this_layer->name = layer_config->name;
  this_layer->layer_in = bottom_layer->layer_out;

  if (this_layer->paral_config != PARAL_TYPE_DATA) {
    PERR("paral type %d not supported in FC layer", this_layer->paral_config);
    exit(-1);
  } else {
    /** Alloate weight and bias*/
    uint nr_imgs = this_layer->layer_in->global_dim.dims[0];
    uint nr_in_flat_dim =
        dim_get_capacity(this_layer->layer_in->global_dim) / nr_imgs;
    uint w_shape[] = {nr_in_flat_dim, layer_config->nr_classes, 0, 0};
    Blob *weight_blob = new Blob(this_layer->name + ".weight", 1, w_shape,
                                 DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight_blob);

    uint b_shape[] = {layer_config->nr_classes, 0, 0, 0};
    Blob *bias_blob = new Blob(this_layer->name + ".bias", 1, b_shape,
                               DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(bias_blob);

    /* Weight init, TODO: need to copy to other others*/
    AWNN_CHECK_EQ(
        S_OK, weight_init_fc_kaiming(weight_blob->data[0], bias_blob->data[0]));

    /*Output setup*/
    uint out_shape[] = {nr_imgs, layer_config->nr_classes, 0, 0};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape,
                                     DATA_PARTITIONED_N, this_layer->topo);

    this_layer->tape.insert({"in", this_layer->layer_in});    // save x->0
    this_layer->tape.insert({"weight", weight_blob});         // save w->1
    this_layer->tape.insert({"bias", bias_blob});             // save b -> 3
    this_layer->tape.insert({"out", this_layer->layer_out});  // save y -> 5
  }
}

double layer_resblock_forward(tensor_t x, tape_t &tape, tensor_t y,
                              void *layer_config, int id) {
  double reg_loss = 0;
  layer_resblock_config_t *config = (layer_resblock_config_t *)(layer_config);
  int padding = 1;
  int stride = 1;

  tensor_t conv1_w = tape["conv1.weight"]->data[id];
  tensor_t conv1_out = tape["conv1.out"]->data[id];
  do_conv_forward_perimg(x, conv1_w, conv1_out, padding, stride);
  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_forward(conv1_out);
  }

  tensor_t conv2_w = tape["conv2.weight"]->data[id];
  do_conv_forward_perimg(conv1_out, conv2_w, y, padding, stride);

  tensor_elemwise_op_inplace(y, x, TENSOR_OP_ADD);

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_forward(y);
  }

  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(conv1_w);
  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(conv2_w);
  return reg_loss;
}

void layer_resblock_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                             void *layer_config, int id) {
  layer_resblock_config_t *config = (layer_resblock_config_t *)(layer_config);
  int padding = 1, stride = 1;
  tensor_t x = tape["in"]->data[id];
  tensor_t dx_iden = tensor_make_alike(x);

  tensor_t conv1_w = tape["conv1.weight"]->data[id];
  tensor_t conv1_dw = tape["conv1.weight"]->diff[id];
  tensor_t conv1_out = tape["conv1.out"]->data[id];
  tensor_t conv1_dout = tape["conv1.out"]->diff[id];

  tensor_t conv2_w = tape["conv2.weight"]->data[id];
  tensor_t conv2_dw = tape["conv2.weight"]->diff[id];
  tensor_t y = tape["out"]->data[id];

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_backward(dy, y);
  }
  tensor_copy(dx_iden, dy);

  do_conv_backward_perimg(conv1_dout, conv2_dw, dy, conv1_out, conv2_w, padding,
                          stride);

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_backward(conv1_dout, conv1_out);
  }

  do_conv_backward_perimg(dx, conv1_dw, conv1_dout, x, conv1_w, padding,
                          stride);

  tensor_elemwise_op_inplace(dx, dx_iden, TENSOR_OP_ADD);

  if (config->reg > 0) {
    update_regulizer_gradient(conv1_w, conv1_dw, config->reg);
    update_regulizer_gradient(conv2_w, conv2_dw, config->reg);
  }

  tensor_destroy(&dx_iden);
}

void layer_resblock_setup(layer_t *this_layer,
                          layer_resblock_config_t *layer_config,
                          layer_t *bottom_layer) {
  this_layer->forward = &layer_resblock_forward;
  this_layer->backward = &layer_resblock_backward;
  this_layer->layer_type = LAYER_TYPE_RESBLOCK;

  this_layer->name = layer_config->name;

  if (this_layer->paral_config != PARAL_TYPE_DATA) {
    PERR("paral type %d not supported in resblocklayer",
         this_layer->paral_config);
    exit(-1);
  } else {
    this_layer->layer_in = bottom_layer->layer_out;

    /*Allocate weight*/
    uint nr_imgs = this_layer->layer_in->global_dim.dims[0];
    uint in_channels = this_layer->layer_in->global_dim.dims[1];
    uint in_height = this_layer->layer_in->global_dim.dims[2];

    /* Allocate weight for first conv layer*/
    uint out_channels = in_channels;
    uint out_height = in_height;

    uint w1_shape[] = {out_channels, in_channels, layer_config->kernel_size,
                       layer_config->kernel_size};
    Blob *weight1_blob = new Blob(this_layer->name + ".conv1.weight", 1,
                                  w1_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight1_blob);
    AWNN_CHECK_EQ(S_OK, weight_init_kaiming(weight1_blob->data[0]));

    uint conv1_out_shape[] = {nr_imgs, out_channels, out_height, out_height};
    Blob *conv1_out_blob =
        new Blob(this_layer->name + ".conv1.out", 1, conv1_out_shape,
                 DATA_PARTITIONED_N, this_layer->topo);
    this_layer->temp_blobs.push_back(conv1_out_blob);

    /* Allocate weight for second conv layer*/
    uint w2_shape[] = {out_channels, out_channels, layer_config->kernel_size,
                       layer_config->kernel_size};
    Blob *weight2_blob = new Blob(this_layer->name + ".conv2.weight", 1,
                                  w2_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight2_blob);
    AWNN_CHECK_EQ(S_OK, weight_init_kaiming(weight2_blob->data[0]));

    /* Layer out*/
    uint out_shape[] = {nr_imgs, out_channels, out_height, out_height};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape,
                                     DATA_PARTITIONED_N, this_layer->topo);

    /* Set tensor other than input/output*/
    this_layer->tape.insert({"in", this_layer->layer_in});  // save x -> 0

    this_layer->tape.insert({"conv1.weight", weight1_blob});  // save w1 -> 1
    this_layer->tape.insert(
        {"conv1.out", conv1_out_blob});  // save conv1_out ->3

    this_layer->tape.insert({"conv2.weight", weight2_blob});  // save w1 -> 1

    this_layer->tape.insert({"out", this_layer->layer_out});  // save out 9
  }
}

static void _do_layer_setup(layer_t *target_layer, void *layer_config,
                            layer_t *bottom_layer) {
  switch (target_layer->layer_type) {
    case LAYER_TYPE_CONV2D:
      layer_conv2d_setup(target_layer, (layer_conv2d_config_t *)layer_config,
                         bottom_layer);
      break;
    case LAYER_TYPE_DATA:
      layer_data_setup(target_layer, (layer_data_config_t *)layer_config,
                       bottom_layer);
      break;
    case LAYER_TYPE_FC:
      layer_fc_setup(target_layer, (layer_fc_config_t *)layer_config,
                     bottom_layer);
      break;

    case LAYER_TYPE_RELU_INPLACE:
      layer_relu_setup(target_layer, (layer_relu_config_t *)layer_config,
                       bottom_layer);
      break;

    case LAYER_TYPE_RESBLOCK:
      layer_resblock_setup(
          target_layer, (layer_resblock_config_t *)layer_config, bottom_layer);
      break;

    case LAYER_TYPE_POOL:
      layer_pool_setup(target_layer, (layer_pool_config_t *)layer_config,
                       bottom_layer);
      break;

    default:
      PERR("layer type %d not support", target_layer->layer_type);
  }
}

/** Initialize this layer with machine topology and parallel policy*/
layer_t *layer_setup(layer_type_t layer_type, void *layer_config,
                     layer_t *bottom_layer, topo_config_t *topo,
                     paral_config_t paral_config) {
  layer_t *target_layer = new layer_t();
  target_layer->paral_config = paral_config;
  target_layer->topo = topo;
  target_layer->config = layer_config;
  target_layer->layer_type = layer_type;

  _do_layer_setup(target_layer, layer_config, bottom_layer);

  return target_layer;
}

void layer_teardown(layer_t *this_layer) {
  if (this_layer->layer_type != LAYER_TYPE_RELU_INPLACE) {
    delete this_layer->layer_out;
    while (!this_layer->learnables.empty()) {
      Blob *param = this_layer->learnables.back();
      this_layer->learnables.pop_back();
      delete param;
    }
    while (!this_layer->temp_blobs.empty()) {
      Blob *sublayer_out = this_layer->temp_blobs.back();
      this_layer->temp_blobs.pop_back();
      delete sublayer_out;
    }
  }
  delete this_layer;
}

void net_add_layer(net_t *net, layer_t *layer) {
  if (layer->layer_type != LAYER_TYPE_DATA) {
    AWNN_CHECK_EQ(net->layers.back()->layer_out, layer->layer_in);
  }
  net->layers.push_back(layer);
}

void net_teardown(net_t *this_net) {
  while (!this_net->layers.empty()) {
    // PDBG("teardown %s", this_net->layers.back()->name.c_str());
    layer_teardown(this_net->layers.back());
    this_net->layers.pop_back();
  }
}

static void _do_concurrent_forward(concurrent_context *context, size_t i) {
  net_t *this_net = context->net;
  for (auto iter_layer = this_net->layers.begin();
       iter_layer != this_net->layers.end(); ++iter_layer) {
    layer_t *layer = *iter_layer;

    if (layer->layer_type == LAYER_TYPE_DATA) continue;
    if (layer->paral_config && layer->paral_config != PARAL_TYPE_DATA) {
      PERR("Bad paral config");
      exit(-1);
    }
    context->reg_losses[i] =
        layer->forward(layer->layer_in->data[i], layer->tape,
                       layer->layer_out->data[i], layer->config, i);
  }
}

double net_forward(net_t *this_net) {
  PDBG("This is the legacy net forward");
  double reg_loss = 0;
  struct concurrent_context context;
  context.net = this_net;
  context.reg_losses = &reg_loss;

  _do_concurrent_forward(&context, 0);

  return reg_loss;
}

static void _do_concurrent_backward(concurrent_context *context, size_t i) {
  net_t *this_net = context->net;
  for (auto iter_layer = this_net->layers.rbegin();
       iter_layer != this_net->layers.rend(); ++iter_layer) {
    layer_t *layer = *iter_layer;

    if (layer->layer_type == LAYER_TYPE_DATA) continue;
    if (layer->paral_config && layer->paral_config != PARAL_TYPE_DATA) {
      PERR("Bad paral config");
      exit(-1);
    }
    layer->backward(layer->layer_in->diff[i], layer->tape,
                    layer->layer_out->diff[i], layer->config, i);
  }
}

void net_backward(net_t *this_net) {
  PDBG("This is the legacy net backward");
  struct concurrent_context context;
  context.net = this_net;
  _do_concurrent_backward(&context, 0);
}

/** Naive all-reduce between all threads*/
static void _do_allreduce(concurrent_context *context, size_t id) {
  // Accumulate all gradients from worker to main
  net_t *local_model = context->net;
  if (id != 0) {
    for (size_t idx_layer = 0; idx_layer < local_model->layers.size();
         idx_layer++) {
      size_t nr_learnables_this_layer =
          local_model->layers[idx_layer]->learnables.size();
      for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
           idx_param++) {
        Blob *param_local =
            local_model->layers[idx_layer]->learnables[idx_param];

        PDBG("updating %s...", param_local->name.c_str());
        AWNN_CHECK_EQ(param_local->learnable, 1);
        // sgd
        // sgd_update(p_param, learning_rate);
        pthread_mutex_lock(context->ptr_mutex);
        tensor_elemwise_op_inplace((param_local)->diff[0],
                                   (param_local)->diff[id], TENSOR_OP_ADD);
        pthread_mutex_unlock(context->ptr_mutex);
      }
    }
  }

  // main thread get avg
#ifdef DO_AVERAGE
  pthread_barrier_wait(context->ptr_barrier);

  topo_config_t *topo = context->topo;

  int nr_parts = topo ? topo->nr_threads : 1;
  if (id == 0 && nr_parts > 1) {
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

        for (ii = 0; ii < capacity; ii++) {
          dparam.data[ii] /= (nr_parts);
        }
      }
    }
  }
#endif

  // each other thread get a copy
  pthread_barrier_wait(context->ptr_barrier);
  if (id != 0) {
    for (size_t idx_layer = 0; idx_layer < local_model->layers.size();
         idx_layer++) {
      size_t nr_learnables_this_layer =
          local_model->layers[idx_layer]->learnables.size();
      for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
           idx_param++) {
        Blob *param_local =
            local_model->layers[idx_layer]->learnables[idx_param];

        PDBG("Duplicating %s...", (param_local)->name.c_str());
        AWNN_CHECK_EQ((param_local)->learnable, 1);
        tensor_copy((param_local)->diff[id], (param_local)->diff[0]);
      }
    }
  }

  pthread_barrier_wait(context->ptr_barrier);
}

#if 0
void _do_reducegradient(concurrent_context *context, size_t id) {
  // Accumulate all gradients from worker to main
  net_t *local_model = context->net;
  if (id != 0) {
    for (size_t idx_layer = 0; idx_layer < local_model->layers.size();
         idx_layer++) {
      size_t nr_learnables_this_layer =
          local_model->layers[idx_layer]->learnables.size();
      for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
           idx_param++) {
        Blob *param_local =
            local_model->layers[idx_layer]->learnables[idx_param];

        PDBG("updating %s...", param_local->name.c_str());
        AWNN_CHECK_EQ(param_local->learnable, 1);
        // sgd
        // sgd_update(p_param, learning_rate);
        pthread_mutex_lock(context->ptr_mutex);
        tensor_elemwise_op_inplace((param_local)->diff[0],
                                   (param_local)->diff[id], TENSOR_OP_ADD);
        pthread_mutex_unlock(context->ptr_mutex);
      }
    }
  }
}
#endif

// recursively tree-based reduce
void _do_step_reduce(concurrent_context *context, size_t id){
  net_t *this_net = context->net;
  topo_config_t *topo = context->topo;

  int nr_parts = topo ? topo->nr_threads : 1;

  for(int s = 1; s< nr_parts ; s*=2){
    if(id %(2*s) == 0 && id + s < nr_parts){
      size_t from_idx = id + s;
      size_t to_idx = id;
      PDBG("[worker %u]: reduce %u <- %u", id, to_idx, from_idx);

      /*
      for (size_t idx_layer = 0; idx_layer < this_net->layers.size();
           idx_layer++) {
        size_t nr_learnables_this_layer =
            this_net->layers[idx_layer]->learnables.size();
        for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
             idx_param++) {
            Blob *paramblob =
              this_net->layers[idx_layer]->learnables[idx_param];
              */

    }

    pthread_barrier_wait(context->ptr_barrier);
  }
}


void _do_step_bcast(concurrent_context *context, size_t id){
  net_t *this_net = context->net;
  topo_config_t *topo = context->topo;

  int nr_parts = topo ? topo->nr_threads : 1;

  for(int s = Power2RoundUp(nr_parts)/2; s >0 ; s/=2){
    if(id %(2*s) == 0 && id+ s< nr_parts){
      size_t from_idx = id;
      size_t to_idx = id+s;
      PDBG("[worker %u]: copy        %u <- %u", id, to_idx, from_idx);
    }
    pthread_barrier_wait(context->ptr_barrier);
  }
}


void allreduce_hybrid(concurrent_context *context) {
  tensor_t x;
  net_t *this_net = context->net;
  topo_config_t *topo = context->topo;

  int nr_parts = topo ? topo->nr_threads : 1;
  /*
  if(!IsPowerOf2(nr_parts)){
    PERR("need special handling for %d workers", nr_parts);
    exit(-1);
  }
  */
  double learning_rate = context->lr;

  // readdata
  if(context->allreduce_type == ALLREDUCE_BARRIER){
  pthreadpool_parallelize_1d(
      topo->threadpool, (pthreadpool_task_1d_t)_do_allreduce, context, nr_parts,
      PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);
  }
  else if(context->allreduce_type == ALLREDUCE_TREE_WITH_UPDATE){
    // reduce(add) to root
    // See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
      
    pthreadpool_parallelize_1d(
        topo->threadpool, (pthreadpool_task_1d_t)_do_step_reduce, context, nr_parts,
        PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);

          // PINF("reduce gradient for %s", paramblob->name.c_str());

    // average + update
    for (size_t idx_layer = 0; idx_layer < this_net->layers.size();
         idx_layer++) {
      size_t nr_learnables_this_layer =
          this_net->layers[idx_layer]->learnables.size();
      for (size_t idx_param = 0; idx_param < nr_learnables_this_layer;
           idx_param++) {
        Blob *paramblob =
            this_net->layers[idx_layer]->learnables[idx_param];
        PDBG("updating %s...", (paramblob)->name.c_str());
        do_sgd_update_momentum(paramblob->data[0], paramblob->diff[0],
                             paramblob->velocity[0], learning_rate/nr_parts, 0.9);

      }
    }

    // broadcast updated weight

    pthreadpool_parallelize_1d(
        topo->threadpool, (pthreadpool_task_1d_t)_do_step_bcast, context, nr_parts,
        PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);

  }
  else{
    PERR("REDUCE type %d not implemented", context->allreduce_type);
  }
}

/**Legacy sgd for single-thread*/
void net_update_weights(net_t *this_net, double learning_rate) {
  for (auto iter_layer = this_net->layers.begin();
       iter_layer != this_net->layers.end(); ++iter_layer) {
    layer_t *layer = *iter_layer;

    if (layer->layer_type == LAYER_TYPE_DATA) continue;
    for (auto param = layer->learnables.begin();
         param != layer->learnables.end(); ++param) {
      PDBG("updating %s...", (*param)->name.c_str());
      AWNN_CHECK_EQ((*param)->learnable, 1);
      // sgd
      // sgd_update(p_param, learning_rate);
      do_sgd_update_momentum((*param)->data[0], (*param)->diff[0],
                             (*param)->velocity[0], learning_rate, 0.9);
    }
  }
}

/* Reduce*/
static void _do_concurrent_update_weights(concurrent_context *context,
                                          size_t i) {
  net_t *this_net = context->net;
  double learning_rate = context->lr;

  if(context->allreduce_type == ALLREDUCE_BARRIER){
    for (auto iter_layer = this_net->layers.begin();
         iter_layer != this_net->layers.end(); ++iter_layer) {
      layer_t *layer = *iter_layer;

      if (layer->layer_type == LAYER_TYPE_DATA) continue;
      for (auto param = layer->learnables.begin();
           param != layer->learnables.end(); ++param) {
        PDBG("updating %s...", (*param)->name.c_str());
        AWNN_CHECK_EQ((*param)->learnable, 1);
        // sgd
        // sgd_update(p_param, learning_rate);
        do_sgd_update_momentum((*param)->data[i], (*param)->diff[i],
                               (*param)->velocity[i], learning_rate, 0.9);
      }
    }
  }
  else if(context->allreduce_type == ALLREDUCE_TREE_WITH_UPDATE){
    // PWRN("skip updates weight");
  }
  else{
    PERR("REDUCE type %d not implemented", context->allreduce_type);
  }
}

void net_update_weights_hybrid(concurrent_context *context) {
  tensor_t x;
  net_t *this_net = context->net;
  topo_config_t *topo = context->topo;

  int nr_parts = topo ? topo->nr_threads : 1;

  // readdata
  pthreadpool_parallelize_1d(
      topo->threadpool, (pthreadpool_task_1d_t)_do_concurrent_update_weights,
      context, nr_parts, PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);
}

void net_loss(net_t *net, tensor_t x, label_t const *labels, T *ptr_loss,
              int verbose) {
  T classify_loss;
  T reg_loss, total_loss;
  tensor_copy(net->layers[0]->layer_out->data[0], x);

  reg_loss = net_forward(net);

  tensor_t out = net->layers.back()->layer_out->data[0];
  tensor_t dout = net->layers.back()->layer_out->diff[0];
  AWNN_CHECK_EQ(S_OK,
                loss_softmax(out, labels, &classify_loss, MODE_TRAIN, dout));
  net_backward(net);

  total_loss = reg_loss + classify_loss;
  if (verbose) {
    PMAJOR(
        "\t: Forward complete with regulizer loss %.3f(classify %.3f +  reg "
        "%.3f",
        total_loss, classify_loss, reg_loss);
  }
  *ptr_loss = total_loss;
}

static void _do_concurrent_read(concurrent_context *context, size_t i) {
  uint cnt_read = get_train_batch_mt(context->loader, i);

  reader_local_info *reader_info = context->loader->readers_info + i;
  tensor_copy(context->net->layers[0]->layer_out->data[i], reader_info->cur_x);
}

static void _do_concurrent_softmax(concurrent_context *context, size_t i) {
  reader_local_info *reader_info = context->loader->readers_info + i;
  T local_classify_loss = 0;
  label_t const *labels = reader_info->cur_label;

  net_t *net = context->net;
  tensor_t out = net->layers.back()->layer_out->data[i];
  tensor_t dout = net->layers.back()->layer_out->diff[i];
  AWNN_CHECK_EQ(
      S_OK, loss_softmax(out, labels, &local_classify_loss, MODE_TRAIN, dout));
  context->classify_losses[i] = local_classify_loss;
}

void net_loss_hybrid(concurrent_context *context, double *ptr_loss,
                     int verbose) {
  tensor_t x;
  net_t *net = context->net;
  data_loader_t *data_loader = context->loader;
  topo_config_t *topo = context->topo;

  double classify_loss;
  double reg_loss, total_loss;

  int nr_parts = topo ? topo->nr_threads : 1;

  // readdata
  pthreadpool_parallelize_1d(
      topo->threadpool, (pthreadpool_task_1d_t)_do_concurrent_read, context,
      nr_parts, PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);

  // forward
  pthreadpool_parallelize_1d(
      topo->threadpool, (pthreadpool_task_1d_t)_do_concurrent_forward, context,
      nr_parts, PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);

  // softmax
  pthreadpool_parallelize_1d(
      topo->threadpool, (pthreadpool_task_1d_t)_do_concurrent_softmax, context,
      nr_parts, PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);

  // backward
  pthreadpool_parallelize_1d(
      topo->threadpool, (pthreadpool_task_1d_t)_do_concurrent_backward, context,
      nr_parts, PTHREADPOOL_FLAG_DISABLE_DENORMALS /* flags */);

  // TODO: Accumulate loss
  reg_loss = context->reg_losses[0];
  classify_loss = context->classify_losses[0];
  total_loss = reg_loss + classify_loss;
  if (verbose) {
    PMAJOR(
        "\t: Forward complete with regulizer loss %.3f(classify %.3f +  "
        "reg(need to average classify loss) "
        "%.3f",
        total_loss, classify_loss, reg_loss);
  }
  *ptr_loss = total_loss;
}
