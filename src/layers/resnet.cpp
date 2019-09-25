#include "layers/layer_common.hpp"
#include "awnn/tensor.h"
#include "utils/weight_init.h"

// TODO: i had use global varaibles otherwise dimension info will be lost
layer_data_config_t dataconfig;
layer_conv2d_config_t conv_config;
layer_resblock_config_t resblock_config;
layer_pool_config_t pool_config;
layer_fc_config_t fc_config;

void resnet_setup(net_t *net, uint input_shape[], double reg){

  /*Conv layer*/
  dataconfig.name = "data";

  dataconfig.dim.dims[0] = input_shape[0];
  dataconfig.dim.dims[1] = input_shape[1];
  dataconfig.dim.dims[2] = input_shape[2];
  dataconfig.dim.dims[3] = input_shape[3];

  layer_t * data_layer = layer_setup(LAYER_TYPE_DATA, &dataconfig, nullptr);
  net_add_layer(net, data_layer);

  /*Conv layer*/
  conv_config.name = "conv2d";
  conv_config.out_channels = 16;
  conv_config.kernel_size = 3;
  conv_config.reg = reg;
  conv_config.activation = ACTIVATION_RELU;

  layer_t * conv_layer = layer_setup(LAYER_TYPE_CONV2D, &conv_config, data_layer);
  net_add_layer(net, conv_layer);

  /*First residual block*/
  resblock_config.name = "resblock";
  resblock_config.reg = reg;

  layer_t * resblock_layer = layer_setup(LAYER_TYPE_RESBLOCK, &resblock_config, conv_layer);
  net_add_layer(net, resblock_layer);

  /*pool layer*/
  pool_config.name = "pool";

  layer_t * pool_layer = layer_setup(LAYER_TYPE_POOL, &pool_config, resblock_layer);
  net_add_layer(net, pool_layer);

  /*FC layer*/
  fc_config.name = "fc";
  fc_config.nr_classes = 10;
  fc_config.reg = reg;
  fc_config.activation = ACTIVATION_NONE;

  layer_t * fc_layer = layer_setup(LAYER_TYPE_FC, &fc_config, pool_layer);
  net_add_layer(net, fc_layer);
}

void resnet_teardown(net_t *net){
  net_teardown(net);
}
