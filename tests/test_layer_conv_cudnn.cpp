/*
 * Description:
 *
 * Author: Yuankun Fu
 * e-mail: fu121@purdue.edu
 */

#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

namespace {
class LayerConvCUDNNTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerConvCUDNNTest, ConvForwardcudnn) {
  conv_param_t conv_params;

  conv_params.stride=1;
  conv_params.padding=1;

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 3;
  uint nr_filter = 3;

  uint sz_out = 1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;
  EXPECT_EQ(4, sz_out);

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x3x3
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x4x4

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  tensor_t d_x = tensor_make_copy_h2d(x);
  tensor_t d_w = tensor_make_copy_h2d(w);
  tensor_t d_y = tensor_make_copy_h2d(y);

  status_t ret = convolution_forward_cudnn(d_x, d_w, NULL, conv_params, d_y);
  EXPECT_EQ(ret, S_OK);

  // device copy back to host
  tensor_copy_d2h(y, d_y);

  tensor_t y_ref = tensor_make_alike(y);
  double value_list[] = {
      0.02553947,  0.03144079,  0.01900658,  0.00722368,  0.01273026,
      0.00692763,  -0.01332237, -0.01829605, -0.03984868, -0.07407237,
      -0.09432237, -0.07371711, -0.05403947, -0.09183553, -0.10640132,
      -0.07898684, 0.05964474,  0.09219079,  0.09894079,  0.06690789,
      0.10225658,  0.15560526,  0.16413158,  0.10959868,  0.12641447,
      0.18971053,  0.19823684,  0.13091447,  0.08238158,  0.12238816,
      0.12700658,  0.08301316,  0.09375,     0.15294079,  0.178875,
      0.12659211,  0.19178289,  0.30428289,  0.34158553,  0.23749342,
      0.29267763,  0.45349342,  0.49079605,  0.33554605,  0.21880263,
      0.33661184,  0.36041447,  0.24501316,  -0.36098684, -0.56540132,
      -0.57783553, -0.40203947, -0.61821711, -0.96507237, -0.98532237,
      -0.68334868, -0.67079605, -1.04607237, -1.06632237, -0.73876974,
      -0.50877632, -0.79099342, -0.80555921, -0.55646053, 0.28701316,
      0.41619079,  0.42294079,  0.27153947,  0.39215132,  0.56486842,
      0.57339474,  0.36538816,  0.41630921,  0.59897368,  0.6075,
      0.38670395,  0.24153947,  0.34407237,  0.34869079,  0.21943421,
      0.93501316,  1.39778289,  1.42371711,  0.94511842,  1.40251974,
      2.09480921,  2.13211184,  1.414125,    1.50341447,  2.24401974,
      2.28132237,  1.51217763,  0.99185526,  1.47913816,  1.50294079,
      0.99532895};
  tensor_fill_list(y_ref, value_list, array_size(value_list));

  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
  PINF("Cudnn_forward Consistent with expected results");

  // free data
}

TEST_F(LayerConvCUDNNTest, ConvBackwardcudnn) {
  conv_param_t conv_params;

  conv_params.stride=1;
  conv_params.padding=1;

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 3;
  uint nr_filter = 3;

  uint sz_out = 1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;
  EXPECT_EQ(4, sz_out);

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x3x3
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x4x4

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  tensor_t d_x = tensor_make_copy_h2d(x);
  tensor_t d_w = tensor_make_copy_h2d(w);
  tensor_t d_y = tensor_make_copy_h2d(y);

  lcache_t cache;
  make_empty_lcache(&cache);

  status_t ret = convolution_forward_cudnn(d_x, d_w, &cache, conv_params, d_y);
  EXPECT_EQ(ret, S_OK);

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));

  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  tensor_t d_dy = tensor_make_copy_h2d(dy);
  tensor_t d_dx = tensor_make_copy_h2d(x);
  tensor_t d_dw = tensor_make_copy_h2d(w);

  ret = convolution_backward_cudnn(d_dx, d_dw, &cache, conv_params, d_dy);
  EXPECT_EQ(ret, S_OK);

  // device copy back to host
  tensor_copy_d2h(dw, d_dw);
  tensor_copy_d2h(dx, d_dx);

  /* II. Numerical check */
  // I had to make this copy since lambda doesn't allow me to use global
  // variable
  tensor_t x_copy = tensor_make_copy(x);
  tensor_t w_copy = tensor_make_copy(w);

  tensor_t dx_ref = tensor_make_alike(x);
  tensor_t dw_ref = tensor_make_alike(w);

  // evaluate gradient of x
  eval_numerical_gradient(
      [&](tensor_t const in, tensor_t out) {
        convolution_forward(in, w_copy, nullptr, conv_params, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-4);
  PINF("cudnn gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [&](tensor_t const in, tensor_t out) {
        convolution_forward(x_copy, in, nullptr, conv_params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-4);
  PINF("cudnn gradient check of w... is ok");

  EXPECT_EQ(ret, S_OK);

  // free data
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
