#include "layers/layer_common.hpp"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "awnn/loss_softmax.h"

namespace {
class LayerTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerTest, ConvLayers) {
  net_t net;
  /*Data layer*/
  layer_data_config_t *dataconfig = new layer_data_config_t();
  dataconfig->name = "data";
  dataconfig->dim.dims[0] = 6;
  dataconfig->dim.dims[1] = 3;
  dataconfig->dim.dims[2] = 32;
  dataconfig->dim.dims[3] = 32;

  layer_t * data_layer = layer_setup(LAYER_TYPE_DATA, dataconfig, nullptr);
  net_add_layer(&net, data_layer);

  /*Conv layer*/
  layer_conv2d_config_t *conv_config = new layer_conv2d_config_t();
  conv_config->name = "conv2d";
  conv_config->out_channels = 4;
  conv_config->kernel_size = 3;
  layer_t * conv_layer = layer_setup(LAYER_TYPE_CONV2D, conv_config, data_layer);
  net_add_layer(&net, conv_layer);

  /*FC layer*/
  layer_fc_config_t *fc_config = new layer_fc_config_t();
  fc_config->name = "fc";
  fc_config->nr_classes = 10;
  layer_t * fc_layer = layer_setup(LAYER_TYPE_FC, fc_config, data_layer);
  net_add_layer(&net, fc_layer);

  net_forward(&net);
  PMAJOR("Forward complete");

  tensor_t out = net.layers.back()->layer_out->data;
  tensor_t dout = net.layers.back()->layer_out->diff;
  double loss;
  label_t labels[] = {0, 1, 2,4, 6, 8};
  AWNN_CHECK_EQ(S_OK,
                loss_softmax(out, labels, &loss, MODE_TRAIN, dout));
  PMAJOR("softmax complete");
  net_backward(&net);
  PMAJOR("Backward complete");
  net_update_weights(&net, 0.01);

  PMAJOR("Weight updated");

  net_teardown(&net);

  delete dataconfig;
  delete conv_config;
  delete fc_config;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
