#include "layers/layer_common.hpp"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

namespace {
class LayerTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerTest, ConvLayers) {
  /*Data layer*/
  layer_data_config_t *dataconfig = new layer_data_config_t();
  dataconfig->name = "data";
  dataconfig->dim.dims[0] = 2;
  dataconfig->dim.dims[1] = 3;
  dataconfig->dim.dims[2] = 4;
  dataconfig->dim.dims[3] = 5;

  layer_t * data_layer = setup_layer(LAYER_TYPE_DATA, dataconfig, nullptr);


  /*Conv layer*/
  layer_conv2d_config_t *config = new layer_conv2d_config_t();
  config->name = "conv2d";
  config->out_channels = 4;
  config->kernel_size = 32;
  layer_t * conv_layer = setup_layer(LAYER_TYPE_CONV2D, config, data_layer);
  EXPECT_EQ(0,0);

  teardown_layer(conv_layer);
  teardown_layer(data_layer);

  delete dataconfig;
  delete config;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
