#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "layers/layer_common.hpp"
#include "test_util.h"
#include "utils/weight_init.h"

namespace {
class LayerTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerTest, FCNet) {
  net_t net;
  double reg = 0;
  /*Data layer*/
  layer_data_config_t dataconfig;
  ;
  dataconfig.name = "data";

  dataconfig.dim.dims[0] = 3;
  dataconfig.dim.dims[1] = 5;
  dataconfig.dim.dims[2] = 0;
  dataconfig.dim.dims[3] = 0;

  layer_t *data_layer = layer_setup(LAYER_TYPE_DATA, &dataconfig, nullptr);
  net_add_layer(&net, data_layer);

  /*FC layer*/
  layer_fc_config_t fc1_config;
  fc1_config.name = "fc1";
  fc1_config.nr_classes = 50;
  fc1_config.reg = reg;
  fc1_config.activation = ACTIVATION_RELU;

  layer_t *fc1_layer = layer_setup(LAYER_TYPE_FC, &fc1_config, data_layer);
  net_add_layer(&net, fc1_layer);

  /*FC layer*/
  layer_fc_config_t fc2_config;
  fc2_config.name = "fc2";
  fc2_config.nr_classes = 7;
  fc2_config.reg = reg;

  layer_t *fc2_layer = layer_setup(LAYER_TYPE_FC, &fc2_config, fc1_layer);
  net_add_layer(&net, fc2_layer);

  /* Forge some fake input*/
  tensor_t x = tensor_make_linspace(-5.5, 4.5, dataconfig.dim.dims, 2);
  label_t labels[] = {0, 5, 1};
  // fill some init values as in cs231n
  weight_init_linspace(fc1_layer->learnables[0]->data[0], -0.7, 0.3);  // w0
  weight_init_linspace(fc1_layer->learnables[1]->data[0], -0.1, 0.9);  // b0
  weight_init_linspace(fc2_layer->learnables[0]->data[0], -0.3, 0.4);  // w1
  weight_init_linspace(fc2_layer->learnables[1]->data[0], -0.9, 0.1);  // b1

  T loss = 0;

  net_loss(&net, x, labels, &loss);
  EXPECT_NEAR(loss, 2.994112658, 1e-5);

  fc1_config.reg = 1.0;
  fc2_config.reg = 1.0;
  net_loss(&net, x, labels, &loss);
  EXPECT_NEAR(loss, 26.11873099, 1e-5);
  PINF("Forward passed, value checked");

#ifndef AWNN_USE_FLT32
  // Check with numerical gradient
  uint y_shape[] = {1};
  tensor_t dy = tensor_make_ones(y_shape, dim_of_shape(y_shape));
  dy.data[0] = 1.0;  // the y is the loss, no upper layer

  // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
  for (auto this_layer : net.layers) {
    for (auto learnable : this_layer->learnables) {
      tensor_t param = learnable->data;
      tensor_t dparam = learnable->diff;
      tensor_t dparam_ref = tensor_make_alike(param);
      net_t *p_net = &net;
      eval_numerical_gradient(
          [p_net, x, labels](tensor_t const, tensor_t out) {
            T *ptr_loss = &out.data[0];
            net_loss(p_net, x, labels, ptr_loss);
          },
          param, dy, dparam_ref);

      EXPECT_LT(tensor_rel_error(dparam_ref, dparam), 1e-7);
      tensor_destroy(&dparam_ref);
    }
  }
  tensor_destroy(&dy);
#endif

  net_teardown(&net);
  tensor_destroy(&x);
}

TEST_F(LayerTest, ConvNet) {
  net_t net;

  /*Conv layer*/
  layer_data_config_t dataconfig;
  dataconfig.name = "data";

  dataconfig.dim.dims[0] = 6;
  dataconfig.dim.dims[1] = 3;
  dataconfig.dim.dims[2] = 32;
  dataconfig.dim.dims[3] = 32;

  layer_t *data_layer = layer_setup(LAYER_TYPE_DATA, &dataconfig, nullptr);
  net_add_layer(&net, data_layer);

  /*Conv layer*/
  layer_conv2d_config_t conv_config;
  conv_config.name = "conv2d";
  conv_config.out_channels = 4;
  conv_config.kernel_size = 3;
  conv_config.reg = 0.001;

  layer_t *conv_layer =
      layer_setup(LAYER_TYPE_CONV2D, &conv_config, data_layer);
  net_add_layer(&net, conv_layer);

  net_teardown(&net);
}

TEST_F(LayerTest, ResBlock) {
  net_t net;
  double reg = 1;

  uint input_shape[4] = {3, 3, 32, 32};

  resnet_setup(&net, input_shape, reg);
  /* Forge some fake input*/
  tensor_t x = tensor_make_linspace(-0.2, 0.3, input_shape, 4);
  label_t labels[] = {0, 5, 1};
  // fill some init values as in cs231n
  layer_t *conv_layer = net.layers[1];
  layer_t *resblock_layer = net.layers[2];
  layer_t *fc_layer = net.layers[4];
  weight_init_linspace(conv_layer->learnables[0]->data[0], -0.7, 0.3);  // w0
  weight_init_linspace(resblock_layer->learnables[0]->data[0], -0.7,
                       0.3);  // conv1.weight
  weight_init_linspace(resblock_layer->learnables[1]->data[0], -0.7,
                       0.3);  // conv2.weight
  weight_init_linspace(fc_layer->learnables[0]->data[0], -0.7, 0.3);  // w1
  weight_init_linspace(fc_layer->learnables[1]->data[0], -0.7, 0.3);  // w1

  T loss = 0;

  net_loss(&net, x, labels, &loss, 1);
  if (reg < 1e-7)
    EXPECT_NEAR(loss, 14.975702563, 1e-3);
  else  // reg == 1
    EXPECT_NEAR(loss, 335.9764923, 1e-3);

    // Check with numerical gradient
#ifndef AWNN_USE_FLT32
  // Check with numerical gradient
  uint y_shape[] = {1};
  tensor_t dy = tensor_make_ones(y_shape, dim_of_shape(y_shape));
  dy.data[0] = 1.0;  // the y is the loss, no upper layer

  // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
  for (auto this_layer : net.layers) {
    for (auto learnable : this_layer->learnables) {
      tensor_t param = learnable->data;
      tensor_t dparam = learnable->diff;
      tensor_t dparam_ref = tensor_make_alike(param);
      PINF("checking gradient of %s", learnable->name.c_str());

      net_t *p_net = &net;
      eval_numerical_gradient(
          [p_net, x, labels](tensor_t const, tensor_t out) {
            T *ptr_loss = &out.data[0];
            net_loss(p_net, x, labels, ptr_loss);
          },
          param, dy, dparam_ref);

      EXPECT_LT(tensor_rel_error(dparam_ref, dparam), 1e-3);
      tensor_destroy(&dparam_ref);
    }
  }
  tensor_destroy(&dy);
#endif

  resnet_teardown(&net);
  tensor_destroy(&x);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
