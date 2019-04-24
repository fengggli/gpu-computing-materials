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
class LayerConvNNPACKTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerConvNNPACKTest, Forward) {
  conv_param_t conv_params;

  conv_params.stride = 2;
  conv_params.padding = 1;

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 4;
  uint nr_filter = 3;

  uint sz_out =
      1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;
  EXPECT_EQ(2, sz_out);

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x4x4
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x2x2

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  status_t ret = convolution_forward_nnpack(
      x, w, NULL, conv_params,
      y);  // forward function should allocate and populate cache;
  // status_t ret = convolution_forward(x, w, NULL, conv_params, y);// forward
  // function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  tensor_t y_ref = tensor_make_alike(y);
  double value_list[] = {
      0.012401913875598115, -0.009877806404122181, -0.08387191755612806,
      -0.11092160471107837, 0.16027088700772907,   0.166610967979389,
      0.17847626058152372,  0.1800463746779536,    0.30813986013986006,
      0.3430997423629002,   0.4408244387191755,    0.47101435406698555,
      -0.8805358851674642,  -0.9314354066985646,   -1.0912889216047112,
      -1.1469584100110415,  0.6410835480309163,    0.618803827751196,
      0.5448097165991902,   0.5177600294442398,    2.162702981229297,
      2.1690430622009567,   2.1809083548030914,    2.1824784688995216};

  tensor_fill_list(y_ref, value_list, array_size(value_list));

  T rel_err = tensor_rel_error(y_ref, y);
  EXPECT_LT(rel_err, 1e-7);
  PINF("Consistent with expected results");
}

TEST_F(LayerConvNNPACKTest, DISABLED_Backward) {
  conv_param_t params;

  params.stride = 2;
  params.padding = 1;

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 4;
  uint nr_filter = 3;

  uint sz_out = 1 + (sz_img + 2 * params.padding - sz_filter) / params.stride;
  EXPECT_EQ(2, sz_out);

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x4x4
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x2x2

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  lcache_t cache;
  make_empty_lcache(&cache);

  status_t ret = convolution_forward_nnpack(
      x, w, &cache, params,
      y);  // forward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));

  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  ret = convolution_backward_nnpack(
      dx, dw, &cache, params,
      dy);  // backward needs to call free_lcache(cache);
  EXPECT_EQ(ret, S_OK);

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
        convolution_forward(in, w_copy, nullptr, params, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [&](tensor_t const in, tensor_t out) {
        convolution_forward(x_copy, in, nullptr, params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-7);
  PINF("gradient check of w... is ok");

  EXPECT_EQ(ret, S_OK);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
