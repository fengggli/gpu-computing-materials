#include "layer_common.hpp"
#include "awnn/common.h"
#include "awnn/net_resnet.h"
#include "awnn/layer_conv.h"
#include "awnn/layer_fc.h"
#include "awnn/solver.h"
#include "utils/weight_init.h"
#include "awnn/loss_softmax.h"

/** I only need those layers:
 * 1. conv, relu, and conv_relu
 * 2. last fc_layer,
 * 3. residual blocks*/

void layer_data_setup(layer_t *this_layer,
                        layer_data_config_t *layer_config,
                        layer_t *bottom_layer){
  this_layer->layer_type = LAYER_TYPE_DATA;
  AWNN_CHECK_EQ(bottom_layer, nullptr);
  this_layer->name = layer_config->name;
  this_layer->layer_in = nullptr;

  /*Calculate output shape*/
  this_layer->layer_out = new Blob(this_layer->name + ".out", 0, layer_config->dim.dims);
  return;
}

static inline void _do_inplace_relu_forward(tensor_t y){
  for(uint i = 0 ; i < tensor_get_capacity(y); i++){
    T *elem = y.data;
    elem[i] = elem[i]>0? elem[i]: 0;
  }
}

static inline void _do_inplace_relu_backward(tensor_t dx, tensor_t x){
  for(uint i = 0 ; i < tensor_get_capacity(dx); i++){
    T *elem = dx.data;
    elem[i] = x.data[i]>0? elem[i]: 0;
  }
}

// Inplace layer, y and x points to same tensor
double layer_relu_forward(tensor_t x,  std::vector<tensor_t*> &tape, tensor_t y, void* layer_config){
  _do_inplace_relu_forward(y);
  return 0;
}

void layer_relu_backward(tensor_t dx,  std::vector<tensor_t*> &tape, tensor_t dy, void * layer_config){
  tensor_t x = *tape[0];
  _do_inplace_relu_backward(dx, x);
}

void layer_relu_setup(layer_t *this_layer,
                        layer_relu_config_t *layer_config,
                        layer_t *bottom_layer){
  this_layer->forward = &layer_relu_forward;
  this_layer->backward = &layer_relu_backward;
  this_layer->layer_type = LAYER_TYPE_RELU_INPLACE;

  this_layer->name = layer_config->name;
  this_layer->layer_in = bottom_layer->layer_out;
  this_layer->layer_out = bottom_layer->layer_out;

  this_layer->tape.push_back(&(this_layer->layer_in->data)); // save x->0
}

double layer_conv2d_forward(tensor_t x,  std::vector<tensor_t*> &tape, tensor_t y, void *layer_config){
  double reg_loss = 0;
  layer_conv2d_config_t * config = (layer_conv2d_config_t *)(layer_config);
  tensor_t w = *(tape[1]);
  do_conv_forward_perimg(x, w, y, config->padding, config->stride);

  if(config->activation == ACTIVATION_RELU){
    _do_inplace_relu_forward(y);
  }

  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(w);
  return reg_loss;
}

void layer_conv2d_backward(tensor_t dx,  std::vector<tensor_t*> &tape, tensor_t dy, void * layer_config){
  layer_conv2d_config_t * config = (layer_conv2d_config_t *)(layer_config);
  tensor_t x = *(tape[0]);
  tensor_t w = *(tape[1]);
  tensor_t dw = *(tape[2]);
  tensor_t y = *(tape[3]);

  if(config->activation == ACTIVATION_RELU){
      _do_inplace_relu_backward(dy, y);
  }

  do_conv_backward_perimg(dx, dw, dy, x, w, config->padding, config->stride);

  if(config->reg>0)
    update_regulizer_gradient(w, dw, config->reg);
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
  this_layer->layer_in = bottom_layer->layer_out;

  uint nr_imgs = this_layer->layer_in->data.dim.dims[0];
  uint in_channels = this_layer->layer_in->data.dim.dims[1];
  uint in_height = this_layer->layer_in->data.dim.dims[2];

  /* Allocate weight*/
  uint w_shape[] = {layer_config->out_channels, in_channels,
                    layer_config->kernel_size, layer_config->kernel_size};
  Blob *weight_blob = new Blob(this_layer->name + ".weight", 1, w_shape);
  this_layer->learnables.push_back(weight_blob);

  /* Weight init*/
  AWNN_CHECK_EQ(S_OK, weight_init_kaiming(weight_blob->data));

  /*Calculate output shape*/
  uint out_height =
      1 + (in_height + 2 * layer_config->padding - layer_config->kernel_size) /
              layer_config->stride;
  uint out_shape[] = {nr_imgs, layer_config->out_channels, out_height,
                      out_height};
  this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape);

  /* Set tensor other than input/output*/
  this_layer->tape.push_back(&(this_layer->layer_in->data)); // save x
  this_layer->tape.push_back(&(weight_blob->data)); //save w
  this_layer->tape.push_back(&(weight_blob->diff)); //save dw
  this_layer->tape.push_back(&(this_layer->layer_out->data)); // save y

  return;
}

double layer_fc_forward(tensor_t x,  std::vector<tensor_t*> &tape, tensor_t y, void* layer_config){
  double reg_loss = 0;
  layer_fc_config_t * config = (layer_fc_config_t *)(layer_config);
  
  tensor_t w = *tape[1];
  tensor_t b = *tape[3];
  do_layer_fc_forward(x, w, b, y);

  if(config->activation == ACTIVATION_RELU){
    _do_inplace_relu_forward(y);
  }

  reg_loss += 0.5 * (config->reg) * tensor_sum_of_square(w);
  // PINF("regloss %.3f", reg_loss);
  return reg_loss;
}

void layer_fc_backward(tensor_t dx,  std::vector<tensor_t*> &tape, tensor_t dy, void * layer_config){
  layer_fc_config_t * config = (layer_fc_config_t *)(layer_config);

  tensor_t dw = *tape[2];
  tensor_t db = *tape[4];
  tensor_t x = *tape[0];
  tensor_t w = *tape[1];
  tensor_t y = *tape[5];

  if(config->activation == ACTIVATION_RELU){
    _do_inplace_relu_backward(dy, y);
  }

  do_layer_fc_backward(dx, dw, db, dy, x, w);

  if(config->reg>0)
    update_regulizer_gradient(w, dw, config->reg);
}

void layer_fc_setup(layer_t *this_layer,
                        layer_fc_config_t *layer_config,
                        layer_t *bottom_layer) {
  this_layer->forward = &layer_fc_forward;
  this_layer->backward = &layer_fc_backward;
  AWNN_CHECK_GT(layer_config->nr_classes, 0);
  this_layer->layer_type = LAYER_TYPE_FC;

  this_layer->name = layer_config->name;
  this_layer->layer_in = bottom_layer->layer_out;

  /** Alloate weight and bias*/
  uint nr_imgs = this_layer->layer_in->data.dim.dims[0];
  uint nr_in_flat_dim = tensor_get_capacity(this_layer->layer_in->data)/ nr_imgs;
  uint w_shape[] = {nr_in_flat_dim, layer_config->nr_classes, 0, 0};
  Blob* weight_blob = new Blob(this_layer->name + ".weight", 1, w_shape);
  this_layer->learnables.push_back(weight_blob);

  uint b_shape[] = {layer_config->nr_classes, 0, 0, 0};
  Blob* bias_blob = new Blob(this_layer->name + ".bias", 1, b_shape);
  this_layer->learnables.push_back(bias_blob);

  /* Weight init*/
  AWNN_CHECK_EQ(S_OK, weight_init_fc_kaiming(weight_blob->data, bias_blob->data));

  /*Output setup*/
  uint out_shape[] = {nr_imgs, layer_config->nr_classes, 0, 0};
  this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape);

  this_layer->tape.push_back(&(this_layer->layer_in->data)); // save x->0
  this_layer->tape.push_back(&(weight_blob->data)); //save w->1
  this_layer->tape.push_back(&(weight_blob->diff)); //save dw->2
  this_layer->tape.push_back(&(bias_blob->data)); //save b -> 3
  this_layer->tape.push_back(&(bias_blob->diff)); //save db->4
  this_layer->tape.push_back(&(this_layer->layer_out->data)); //save y -> 5
  this_layer->tape.push_back(&(this_layer->layer_out->data)); //save y -> 3
}



layer_t *layer_setup(layer_type_t layer_type, void *layer_config,
                     layer_t *bottom_layer) {
  layer_t *target_layer = new layer_t();
  target_layer->config = layer_config;
  target_layer->layer_type = layer_type;
  switch (layer_type) {
    case LAYER_TYPE_CONV2D:
      layer_conv2d_setup(target_layer, (layer_conv2d_config_t *)layer_config,
                         bottom_layer);
      break;
    case LAYER_TYPE_DATA:
      layer_data_setup(target_layer, (layer_data_config_t *)layer_config, bottom_layer);
      break;
    case LAYER_TYPE_FC:
      layer_fc_setup(target_layer, (layer_fc_config_t *)layer_config, bottom_layer);
      break;

    case LAYER_TYPE_RELU_INPLACE:
      layer_relu_setup(target_layer,(layer_relu_config_t *)layer_config,  bottom_layer);
      break;


    default:
      PERR("layer type %d not support", layer_type);
  }
  return target_layer;
}

void layer_teardown(layer_t * this_layer){
  if(this_layer->layer_type != LAYER_TYPE_RELU_INPLACE){
    delete this_layer->layer_out;
    while(!this_layer->learnables.empty()){
      Blob *param = this_layer->learnables.back();
      this_layer->learnables.pop_back();
      delete param;
    }
  }
  delete this_layer;
}

void net_add_layer(net_t *net, layer_t *layer){
  if(layer->layer_type != LAYER_TYPE_DATA){
    AWNN_CHECK_EQ(net->layers.back()->layer_out, layer->layer_in);
  }
  net->layers.push_back(layer);
}

void net_teardown(net_t *this_net){
  while(!this_net->layers.empty()){
    // PDBG("teardown %s", this_net->layers.back()->name.c_str());
    layer_teardown(this_net->layers.back());
    this_net->layers.pop_back();
  }
}

double net_forward(net_t *this_net){
  double reg_loss = 0;
  for(auto iter_layer = this_net->layers.begin(); iter_layer!= this_net->layers.end(); ++iter_layer){

    layer_t* layer = *iter_layer;

    if(layer->layer_type == LAYER_TYPE_DATA) continue;
    reg_loss += layer->forward(layer->layer_in->data, layer->tape, layer->layer_out->data, layer->config);
  }
  return reg_loss;
}

void net_backward(net_t *this_net){
  for(auto iter_layer = this_net->layers.rbegin(); iter_layer!= this_net->layers.rend(); ++iter_layer){
    layer_t* layer = *iter_layer;

    if(layer->layer_type == LAYER_TYPE_DATA) continue;
    layer->backward(layer->layer_in->diff, layer->tape, layer->layer_out->diff, layer->config);
  }
}

void net_update_weights(net_t * this_net, double learning_rate){
  for(auto iter_layer = this_net->layers.begin(); iter_layer!= this_net->layers.end(); ++iter_layer){
    layer_t* layer = *iter_layer;

    if(layer->layer_type == LAYER_TYPE_DATA) continue;
    for(auto param = layer->learnables.begin(); param != layer->learnables.end(); ++param){
      PDBG("updating %s...", (*param)->name.c_str());
      AWNN_CHECK_EQ((*param)->learnable, 1);
      // sgd
      // sgd_update(p_param, learning_rate);
     do_sgd_update_momentum((*param)->data, (*param)->diff, (*param)->velocity, learning_rate, 0.9);
    }
  }
}

void net_loss(net_t *net, tensor_t x, label_t const *labels,
                  T *ptr_loss){

    double classify_loss, reg_loss, total_loss;
    tensor_copy(net->layers[0]->layer_out->data, x);

    reg_loss = net_forward(net);

    tensor_t out = net->layers.back()->layer_out->data;
    tensor_t dout = net->layers.back()->layer_out->diff;
    AWNN_CHECK_EQ(S_OK,
                  loss_softmax(out, labels, &classify_loss, MODE_TRAIN, dout));
    net_backward(net);

    total_loss = reg_loss + classify_loss;
    /*PMAJOR("\t: Forward complete with regulizer loss %.3f(classify %.3f +  reg %.3f", */
        /*total_loss, classify_loss, reg_loss);*/
    *ptr_loss = total_loss;
}
