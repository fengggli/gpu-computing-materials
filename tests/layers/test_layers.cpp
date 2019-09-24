#include "layers/layer_common.hpp"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "utils/weight_init.h"

namespace {
class LayerTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerTest, FCNet) {
  net_t net;
  double reg = 0;
  /*Data layer*/
  layer_data_config_t dataconfig;;
  dataconfig.name = "data";

  dataconfig.dim.dims[0] = 3;
  dataconfig.dim.dims[1] = 5;
  dataconfig.dim.dims[2] = 0;
  dataconfig.dim.dims[3] = 0;

  layer_t * data_layer = layer_setup(LAYER_TYPE_DATA, &dataconfig, nullptr);
  net_add_layer(&net, data_layer);
  
  /*FC layer*/
  layer_fc_config_t fc1_config;
  fc1_config.name = "fc1";
  fc1_config.nr_classes = 50;
  fc1_config.reg = reg;
  fc1_config.activation = ACTIVATION_RELU;

  layer_t * fc1_layer = layer_setup(LAYER_TYPE_FC, &fc1_config, data_layer);
  net_add_layer(&net, fc1_layer);

  /*FC layer*/
  layer_fc_config_t fc2_config;
  fc2_config.name = "fc2";
  fc2_config.nr_classes = 7;
  fc2_config.reg = reg;

  layer_t * fc2_layer = layer_setup(LAYER_TYPE_FC, &fc2_config, fc1_layer);
  net_add_layer(&net, fc2_layer);

  /* Forge some fake input*/
  tensor_t x = tensor_make_linspace(-5.5, 4.5, dataconfig.dim.dims, 2);
  label_t labels[] = {0, 5, 1};
  // fill some init values as in cs231n
  weight_init_linspace(fc1_layer->learnables[0]->data, -0.7, 0.3); //w0
  weight_init_linspace(fc1_layer->learnables[1]->data, -0.1, 0.9); //b0
  weight_init_linspace(fc2_layer->learnables[0]->data, -0.3, 0.4); //w1
  weight_init_linspace(fc2_layer->learnables[1]->data, -0.9, 0.1); //b1

  double loss = 0;

  net_loss(&net, x, labels, &loss);
  EXPECT_NEAR(loss, 2.994112658, 1e-7);

  fc1_config.reg = 1.0;
  fc2_config.reg = 1.0;
  net_loss(&net, x, labels, &loss);
  EXPECT_NEAR(loss, 26.11873099, 1e-7);
  PINF("Forward passed, value checked");

  // Check with numerical gradient
  uint y_shape[] = {1};
  tensor_t dy = tensor_make_ones(y_shape, dim_of_shape(y_shape));
  dy.data[0] = 1.0;  // the y is the loss, no upper layer

  // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
  for(auto this_layer : net.layers){
    for(auto learnable : this_layer->learnables){
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

  layer_t * data_layer = layer_setup(LAYER_TYPE_DATA, &dataconfig, nullptr);
  net_add_layer(&net, data_layer);

  /*Conv layer*/
  layer_conv2d_config_t conv_config;
  conv_config.name = "conv2d";
  conv_config.out_channels = 4;
  conv_config.kernel_size = 3;
  conv_config.reg = 0.001;

  layer_t * conv_layer = layer_setup(LAYER_TYPE_CONV2D, &conv_config, data_layer);
  net_add_layer(&net, conv_layer);


  net_teardown(&net);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
