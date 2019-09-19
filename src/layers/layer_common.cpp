#include "layer_common.hpp"
#include "awnn/common.h"
#include "awnn/net_resnet.h"

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
  this_layer->learnables.push_back(new Blob(this_layer->name + ".weight", 1, w_shape));

  /*Calculate output shape*/
  uint out_height =
      1 + (in_height + 2 * layer_config->padding - layer_config->kernel_size) /
              layer_config->stride;
  uint out_shape[] = {nr_imgs, layer_config->out_channels, out_height,
                      out_height};
  this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape);

  /* Setup Forward/backward*/

  /* TODO: Workerspace buffer*/
  return;
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
  this_layer->learnables.push_back(new Blob(this_layer->name + ".weight", 1, w_shape));

  uint b_shape[] = {layer_config->nr_classes, 0, 0, 0};
  this_layer->learnables.push_back(new Blob(this_layer->name + ".bias", 1, b_shape));

  /*Output setup*/
  uint out_shape[] = {nr_imgs, layer_config->nr_classes, 0, 0};

  this_layer->layer_out = new Blob(this_layer->name + ".out", 0, out_shape);
}

layer_t *setup_layer(layer_type_t layer_type, void *layer_config,
                     layer_t *bottom_layer) {
  layer_t *target_layer = new layer_t();
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

void teardown_layer(layer_t * this_layer){

  delete this_layer->layer_out;
  while(!this_layer->learnables.empty()){
    Blob *param = this_layer->learnables.back();
    this_layer->learnables.pop_back();
    delete param;
  }
  delete this_layer;
}
