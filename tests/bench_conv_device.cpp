/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

namespace {
class LayerBenchConvDeviceTest : public ::testing::Test {};
}  // namespace



TEST_F(LayerBenchConvDeviceTest, BenchCUDNN) {
  conv_param_t conv_params;

  conv_params.stride = 1;
  conv_params.padding = 1;

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 3;
  uint nr_filter = 3;

  uint sz_out =
      1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;
  EXPECT_EQ(4, sz_out);

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x3x3
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x4x4

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));

  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  lcache_t cache;
  make_empty_lcache(&cache);


  status_t ret = convolution_forward_cudnn(x, w, &cache, conv_params, y);
  EXPECT_EQ(ret, S_OK);

  ret = convolution_backward_cudnn(dx, dw, &cache, conv_params, dy);
  EXPECT_EQ(ret, S_OK);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

