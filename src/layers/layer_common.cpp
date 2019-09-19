#include "layer_common.hpp"
#include "awnn/common.h"
#include "awnn/net_resnet.h"
#include "awnn/layer_conv.h"

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

void layer_conv2d_forward(tensor_t x,  std::vector<tensor_t*> &tape, tensor_t y, void *layer_config){
  layer_conv2d_config_t * config = (layer_conv2d_config_t *)(layer_config);
  tensor_t w = *(tape[1]);
  do_conv_forward_perimg(x, w, y, config->padding, config->stride);
}

void layer_conv2d_backward(tensor_t dx,  std::vector<tensor_t*> &tape, tensor_t dy, void * layer_config){
  layer_conv2d_config_t * config = (layer_conv2d_config_t *)(layer_config);
  tensor_t x = *(tape[0]);
  tensor_t w = *(tape[1]);
  tensor_t dw = *(tape[2]);
  do_conv_backward_perimg(dx, dw, dy, x, w, config->padding, config->stride);
}

void layer_conv2d_setup(layer_t *this_layer,
                        layer_conv2d_config_t *layer_config,
                        layer_t *bottom_layer) {
  this_layer->layer_type = LAYER_TYPE_CONV2D;
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

  /*Calculate output shape*/
  uint out_height =
      1 + (in_height + 2 * layer_config->padding - layer_config->kernel_size) /
              layer_config->stride;
  uint out_shape[] = {nr_imgs, layer_config->out_channels, out_height,
                      out_height};
  this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape);

  /* Setup Forward/backward*/
  this_layer->tape.push_back(&(this_layer->layer_in->data)); // save x
  this_layer->tape.push_back(&(weight_blob->data)); //save w
  this_layer->tape.push_back(&(weight_blob->diff)); //save dw

  /* TODO: Workerspace buffer*/
  return;
}

status_t layer_fc_forward(tensor_t x,  std::vector<tensor_t*> &tape, tensor_t y, void* layer_config){
}
status_t layer_fc_backward(tensor_t dx,  std::vector<tensor_t*> &tape, tensor_t dy, void * layer_config){
}


void layer_fc_setup(layer_t *this_layer,
                        layer_fc_config_t *layer_config,
                        layer_t *bottom_layer) {
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

  /*Output setup*/
  uint out_shape[] = {nr_imgs, layer_config->nr_classes, 0, 0};
  this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape);

  this_layer->tape.push_back(&(this_layer->layer_in->data)); // save x
  this_layer->tape.push_back(&(weight_blob->data)); //save w
  this_layer->tape.push_back(&(weight_blob->diff)); //save dw

}

layer_t *layer_setup(layer_type_t layer_type, void *layer_config,
                     layer_t *bottom_layer) {
  layer_t *target_layer = new layer_t();
  target_layer->config = layer_config;
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

    default:
      PERR("layer type %d not support", layer_type);
  }
  return target_layer;
}

void layer_teardown(layer_t * this_layer){

  delete this_layer->layer_out;
  while(!this_layer->learnables.empty()){
    Blob *param = this_layer->learnables.back();
    this_layer->learnables.pop_back();
    delete param;
  }
  delete this_layer;
}

void net_add_layer(net_t *net, layer_t *layer){
  net->layers.push_back(layer);
}

void net_teardown(net_t *this_net){
  while(!this_net->layers.empty()){
    layer_teardown(this_net->layers.back());
    this_net->layers.pop_back();
  }
}

void net_forward(net_t *this_net){
  for(auto iter_layer = this_net->layers.begin(); iter_layer!= this_net->layers.end(); ++iter_layer){
    layer_t* layer = *iter_layer;

    if(layer->layer_type == LAYER_TYPE_DATA) continue;
    layer->forward(layer->layer_in->data, layer->tape, layer->layer_out->data, layer->config);

  }
}

void net_backward(net_t *this_net){
  for(auto iter_layer = this_net->layers.rbegin(); iter_layer!= this_net->layers.rend(); ++iter_layer){
    layer_t* layer = *iter_layer;

    if(layer->layer_type == LAYER_TYPE_DATA) continue;
    layer->backward(layer->layer_in->data, layer->tape, layer->layer_out->data, layer->config);
  }
}
