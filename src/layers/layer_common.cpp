#include "layer_common.hpp"
#include "awnn/common.h"
#include "awnn/layer_conv.h"
#include "awnn/layer_fc.h"
#include "awnn/layer_pool.h"
#include "awnn/loss_softmax.h"
#include "awnn/net_resnet.h"
#include "awnn/solver.h"
#include "utils/weight_init.h"

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
  if(this_layer->paral_config && *(this_layer->paral_config) != PARAL_TYPE_DATA){
    PERR("paral type %d not supported in FC layer", *(this_layer->paral_config));
    exit(-1);
  }
  else{
    this_layer->layer_out =
        new Blob(this_layer->name + ".out", 0, layer_config->dim.dims, DATA_PARTITIONED_N, this_layer->topo);
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
                          void *layer_config) {
  _do_inplace_relu_forward(y);
  return 0;
}

void layer_relu_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                         void *layer_config) {
  tensor_t x = tape["in"]->data[0];
  _do_inplace_relu_backward(dx, x);
}

void layer_relu_setup(layer_t *this_layer, layer_relu_config_t *layer_config,
                      layer_t *bottom_layer) {
  this_layer->forward = &layer_relu_forward;
  this_layer->backward = &layer_relu_backward;
  this_layer->layer_type = LAYER_TYPE_RELU_INPLACE;

  this_layer->name = layer_config->name;

  if(this_layer->paral_config && *(this_layer->paral_config) != PARAL_TYPE_DATA){
    PERR("paral type %d not supported in FC layer", *(this_layer->paral_config));
    exit(-1);
  }
  else{
    this_layer->layer_in = bottom_layer->layer_out;
    this_layer->layer_out = bottom_layer->layer_out;

    this_layer->tape.insert({"in", this_layer->layer_in});
  }
}

// global averge pool layer
double layer_pool_forward(tensor_t x, tape_t &tape, tensor_t y,
                          void *layer_config) {
  do_global_pool_forward(x, y);
  return 0;
}

void layer_pool_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                         void *layer_config) {
  do_global_pool_backward(dx, dy);
}

void layer_pool_setup(layer_t *this_layer, layer_pool_config_t *layer_config,
                      layer_t *bottom_layer) {
  this_layer->forward = &layer_pool_forward;
  this_layer->backward = &layer_pool_backward;
  this_layer->layer_type = LAYER_TYPE_POOL;

  this_layer->name = layer_config->name;

  if(this_layer->paral_config && *(this_layer->paral_config) != PARAL_TYPE_DATA){
    PERR("paral type %d not supported in FC layer", *(this_layer->paral_config));
    exit(-1);
  }
  else{
    this_layer->layer_in = bottom_layer->layer_out;

    uint nr_imgs = this_layer->layer_in->data[0].dim.dims[0];
    uint in_channels = this_layer->layer_in->data[0].dim.dims[1];

    /*Calculate output shape*/
    uint out_shape[] = {nr_imgs, in_channels, 1, 1};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape, DATA_PARTITIONED_N, this_layer->topo);
  }
}

double layer_conv2d_forward(tensor_t x, tape_t &tape, tensor_t y,
                            void *layer_config) {
  double reg_loss = 0;
  layer_conv2d_config_t *config = (layer_conv2d_config_t *)(layer_config);
  tensor_t w = tape["weight"]->data[0];
  do_conv_forward_perimg(x, w, y, config->padding, config->stride);

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_forward(y);
  }

  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(w);
  return reg_loss;
}

void layer_conv2d_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                           void *layer_config) {
  layer_conv2d_config_t *config = (layer_conv2d_config_t *)(layer_config);
  tensor_t x = tape["in"]->data[0];
  tensor_t w = tape["weight"]->data[0];
  tensor_t dw = tape["weight"]->diff[0];
  tensor_t y = tape["out"]->data[0];

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

  if(this_layer->paral_config && *(this_layer->paral_config) != PARAL_TYPE_DATA){
    PERR("paral type %d not supported in FC layer", *(this_layer->paral_config));
    exit(-1);
  }
  else{

    this_layer->layer_in = bottom_layer->layer_out;

    uint nr_imgs = this_layer->layer_in->data[0].dim.dims[0];
    uint in_channels = this_layer->layer_in->data[0].dim.dims[1];
    uint in_height = this_layer->layer_in->data[0].dim.dims[2];

    /* Allocate weight*/
    uint w_shape[] = {layer_config->out_channels, in_channels,
                      layer_config->kernel_size, layer_config->kernel_size};
    Blob *weight_blob = new Blob(this_layer->name + ".weight", 1, w_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight_blob);

    /* Weight init*/
    AWNN_CHECK_EQ(S_OK, weight_init_kaiming(weight_blob->data[0]));

    /*Calculate output shape*/
    uint out_height =
        1 + (in_height + 2 * layer_config->padding - layer_config->kernel_size) /
                layer_config->stride;
    uint out_shape[] = {nr_imgs, layer_config->out_channels, out_height,
                        out_height};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape, DATA_PARTITIONED_N, this_layer->topo);

    /* Set tensor other than input/output*/
    this_layer->tape.insert({"in", this_layer->layer_in});    // save x
    this_layer->tape.insert({"weight", weight_blob});         // save w
    this_layer->tape.insert({"out", this_layer->layer_out});  // save y

    return;
  }
}

double layer_fc_forward(tensor_t x, tape_t &tape, tensor_t y,
                        void *layer_config) {
  double reg_loss = 0;
  layer_fc_config_t *config = (layer_fc_config_t *)(layer_config);

  tensor_t w = tape["weight"]->data[0];
  tensor_t b = tape["bias"]->data[0];
  do_layer_fc_forward(x, w, b, y);

  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_forward(y);
  }

  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(w);
  // PINF("regloss %.3f", reg_loss);
  return reg_loss;
}

void layer_fc_backward(tensor_t dx, tape_t &tape, tensor_t dy,
                       void *layer_config) {
  layer_fc_config_t *config = (layer_fc_config_t *)(layer_config);

  tensor_t dw = tape["weight"]->diff[0];
  tensor_t db = tape["bias"]->diff[0];
  tensor_t x = tape["in"]->data[0];
  tensor_t w = tape["weight"]->data[0];
  tensor_t y = tape["out"]->data[0];

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

  if(this_layer->paral_config && *(this_layer->paral_config) != PARAL_TYPE_DATA){
    PERR("paral type %d not supported in FC layer", *(this_layer->paral_config));
    exit(-1);
  }
  else{
    /** Alloate weight and bias*/
    uint nr_imgs = this_layer->layer_in->data[0].dim.dims[0];
    uint nr_in_flat_dim =
        tensor_get_capacity(this_layer->layer_in->data[0]) / nr_imgs;
    uint w_shape[] = {nr_in_flat_dim, layer_config->nr_classes, 0, 0};
    Blob *weight_blob = new Blob(this_layer->name + ".weight", 1, w_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight_blob);

    uint b_shape[] = {layer_config->nr_classes, 0, 0, 0};
    Blob *bias_blob = new Blob(this_layer->name + ".bias", 1, b_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(bias_blob);

    /* Weight init, TODO: need to copy to other others*/
    AWNN_CHECK_EQ(S_OK,
                  weight_init_fc_kaiming(weight_blob->data[0], bias_blob->data[0]));

    /*Output setup*/
    uint out_shape[] = {nr_imgs, layer_config->nr_classes, 0, 0};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape, DATA_PARTITIONED_N, this_layer->topo);

    this_layer->tape.insert({"in", this_layer->layer_in});    // save x->0
    this_layer->tape.insert({"weight", weight_blob});         // save w->1
    this_layer->tape.insert({"bias", bias_blob});             // save b -> 3
    this_layer->tape.insert({"out", this_layer->layer_out});  // save y -> 5
  }
  
}

double layer_resblock_forward(tensor_t x, tape_t &tape, tensor_t y,
                              void *layer_config) {
  double reg_loss = 0;
  layer_resblock_config_t *config = (layer_resblock_config_t *)(layer_config);
  int padding = 1;
  int stride = 1;

  tensor_t conv1_w = tape["conv1.weight"]->data[0];
  tensor_t conv1_out = tape["conv1.out"]->data[0];
  do_conv_forward_perimg(x, conv1_w, conv1_out, padding, stride);
  if (config->activation == ACTIVATION_RELU) {
    _do_inplace_relu_forward(conv1_out);
  }

  tensor_t conv2_w = tape["conv2.weight"]->data[0];
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
                             void *layer_config) {
  layer_resblock_config_t *config = (layer_resblock_config_t *)(layer_config);
  int padding = 1, stride = 1;
  tensor_t x = tape["in"]->data[0];
  tensor_t dx_iden = tensor_make_alike(x);

  tensor_t conv1_w = tape["conv1.weight"]->data[0];
  tensor_t conv1_dw = tape["conv1.weight"]->diff[0];
  tensor_t conv1_out = tape["conv1.out"]->data[0];
  tensor_t conv1_dout = tape["conv1.out"]->diff[0];

  tensor_t conv2_w = tape["conv2.weight"]->data[0];
  tensor_t conv2_dw = tape["conv2.weight"]->diff[0];
  tensor_t y = tape["out"]->data[0];

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

  if(this_layer->paral_config && *(this_layer->paral_config) != PARAL_TYPE_DATA){
    PERR("paral type %d not supported in FC layer", *(this_layer->paral_config));
    exit(-1);
  }
  else{

    this_layer->layer_in = bottom_layer->layer_out;

    /*Allocate weight*/
    uint nr_imgs = this_layer->layer_in->data[0].dim.dims[0];
    uint in_channels = this_layer->layer_in->data[0].dim.dims[1];
    uint in_height = this_layer->layer_in->data[0].dim.dims[2];

    /* Allocate weight for first conv layer*/
    uint out_channels = in_channels;
    uint out_height = in_height;

    uint w1_shape[] = {out_channels, in_channels, layer_config->kernel_size,
                       layer_config->kernel_size};
    Blob *weight1_blob =
        new Blob(this_layer->name + ".conv1.weight", 1, w1_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight1_blob);
    AWNN_CHECK_EQ(S_OK, weight_init_kaiming(weight1_blob->data[0]));

    uint conv1_out_shape[] = {nr_imgs, out_channels, out_height, out_height};
    Blob *conv1_out_blob =
        new Blob(this_layer->name + ".conv1.out", 1, conv1_out_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->temp_blobs.push_back(conv1_out_blob);

    /* Allocate weight for second conv layer*/
    uint w2_shape[] = {out_channels, out_channels, layer_config->kernel_size,
                       layer_config->kernel_size};
    Blob *weight2_blob =
        new Blob(this_layer->name + ".conv2.weight", 1, w2_shape, DATA_REPLICATED, this_layer->topo);
    this_layer->learnables.push_back(weight2_blob);
    AWNN_CHECK_EQ(S_OK, weight_init_kaiming(weight2_blob->data[0]));

    /* Layer out*/
    uint out_shape[] = {nr_imgs, out_channels, out_height, out_height};
    this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape, DATA_PARTITIONED_N, this_layer->topo);

    /* Set tensor other than input/output*/
    this_layer->tape.insert({"in", this_layer->layer_in});  // save x -> 0

    this_layer->tape.insert({"conv1.weight", weight1_blob});  // save w1 -> 1
    this_layer->tape.insert({"conv1.out", conv1_out_blob});  // save conv1_out ->3

    this_layer->tape.insert({"conv2.weight", weight2_blob});  // save w1 -> 1

    this_layer->tape.insert({"out", this_layer->layer_out});  // save out 9
  }
}

static void _do_layer_setup(layer_t *target_layer, void * layer_config, layer_t *bottom_layer){
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

layer_t *layer_setup(layer_type_t layer_type, void *layer_config,
                     layer_t *bottom_layer) {
  return layer_setup_hybrid(layer_type, layer_config, bottom_layer, NULL, NULL);
}


/** Initialize this layer with machine topology and parallel policy*/
layer_t *layer_setup_hybrid(layer_type_t layer_type, void *layer_config,
                     layer_t *bottom_layer, topo_config_t *topo,  paral_config_t *paral_config){
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

double net_forward(net_t *this_net) {
  double reg_loss = 0;
  for (auto iter_layer = this_net->layers.begin();
       iter_layer != this_net->layers.end(); ++iter_layer) {
    layer_t *layer = *iter_layer;

    if (layer->layer_type == LAYER_TYPE_DATA) continue;
    reg_loss += layer->forward(layer->layer_in->data[0], layer->tape,
                               layer->layer_out->data[0], layer->config);
  }
  return reg_loss;
}

void net_backward(net_t *this_net) {
  for (auto iter_layer = this_net->layers.rbegin();
       iter_layer != this_net->layers.rend(); ++iter_layer) {
    layer_t *layer = *iter_layer;

    if (layer->layer_type == LAYER_TYPE_DATA) continue;
    layer->backward(layer->layer_in->diff[0], layer->tape, layer->layer_out->diff[0],
                    layer->config);
  }
}

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
      do_sgd_update_momentum((*param)->data[0], (*param)->diff[0], (*param)->velocity[0],
                             learning_rate, 0.9);
    }
  }
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
