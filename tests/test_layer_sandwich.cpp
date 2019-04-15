/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/layer_sandwich.h"
#include "awnn/tensor.h"
#include "config.h"
#include "gtest/gtest.h"
#include "test_util.h"
namespace {

// The fixture for testing class Foo.
class LayerSandwich : public ::testing::Test {
 protected:
  // Objects declared here can be used by all tests in the test case for Foo.
};

TEST_F(LayerSandwich, fc_relu) {}

TEST_F(LayerSandwich, conv_relu) {
  lcache_t cache;

  uint shape_x[] = {2, 3, 8, 8};
  uint shape_w[] = {4, 3, 3, 3};
  uint shape_y[] = {2, 4, 8, 8};

  conv_param_t params;
  params.stride = 1;
  params.padding = 1;

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, array_size(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, array_size(shape_w));

  tensor_t y = tensor_make_zeros(shape_y, array_size(shape_y));

  make_empty_lcache(&cache);
  // forward
  EXPECT_EQ(S_OK, conv_relu_forward(x, w, &cache, params, y));

  // backward
  tensor_t dy = tensor_make_linspace_alike(0.1, 0.5, y);  // make it radom

  // output for backward
  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  EXPECT_EQ(S_OK, conv_relu_backward(dx, dw, &cache, params, dy));

  // numerical check
  tensor_t dx_ref = tensor_make_alike(x);
  tensor_t dw_ref = tensor_make_alike(w);
  
  // evaluate gradient of x
  eval_numerical_gradient(
      [w, params](tensor_t const in, tensor_t out) {
        conv_relu_forward(in, w, NULL,  params, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [x, params](tensor_t const in, tensor_t out) {
        conv_relu_forward(x, in, NULL, params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-7);
  PINF("gradient check of w... is ok");


}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
