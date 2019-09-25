/** Test convolutoin when stride cannot be evenly done*/
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

namespace {

class LayerConvTest : public ::testing::Test {};

TEST_F(LayerConvTest, Backward) {
  conv_param_t params;
  set_conv_method(CONV_METHOD_PERIMG);

  params.stride = 2;
  params.padding = 1;

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 3;
  uint nr_filter = 3;

  uint sz_out = 1 + (sz_img + 2 * params.padding - sz_filter) / params.stride;

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x4x4
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x2x2

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  lcache_t cache;
  make_empty_lcache(&cache);

  status_t ret = convolution_forward(
      x, w, &cache, params,
      y);  // forward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));

  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  ret = convolution_backward(dx, dw, &cache, params,
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
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-3);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [&](tensor_t const in, tensor_t out) {
        convolution_forward(x_copy, in, nullptr, params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-4);
  PINF("gradient check of w... is ok");

  EXPECT_EQ(ret, S_OK);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
