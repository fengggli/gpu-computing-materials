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
//#define TEST_MORE
namespace {

// The fixture for testing class Foo.
class LayerSandwich : public ::testing::Test {
 protected:
  // Objects declared here can be used by all tests in the test case for Foo.
};

TEST_F(LayerSandwich, fc_relu) {}

TEST_F(LayerSandwich, conv_relu) {
  lcache_t cache;

  int shape_x[] = {2, 3, 8, 8};
  int shape_w[] = {4, 3, 3, 3};
  int shape_y[] = {2, 4, 8, 8};

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
        conv_relu_forward(in, w, NULL, params, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-4);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [x, params](tensor_t const in, tensor_t out) {
        conv_relu_forward(x, in, NULL, params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-5);
  PINF("gradient check of w... is ok");
}

TEST_F(LayerSandwich, ResidualBlock_noBN) {
  lcache_t cache;

  int N, C, H, W, F, HH, WW;
#ifdef TEST_MORE
  N = 2, C = 3, H = 5, W = 5;
  F = 3, HH = 3, WW = 3;
#else
  N = 1, C = 1, H = 4, W = 4;
  F = 1, HH = 3, WW = 3;
#endif

  int shape_x[] = {N, C, H, W};
  int shape_w[] = {F, C, HH, WW};
  int shape_y[] = {N, F, H, W};  // stride=1 padding=1 won't change feature H/W

  conv_param_t params;
  params.stride = 1;
  params.padding = 1;

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, array_size(shape_x));
  tensor_t w1 = tensor_make_linspace(-0.2, 0.3, shape_w, array_size(shape_w));
  tensor_t w2 = tensor_make_linspace(-0.2, 0.3, shape_w, array_size(shape_w));

  tensor_t y = tensor_make_zeros(shape_y, array_size(shape_y));
  tensor_t y_ref = tensor_make_zeros(shape_y, array_size(shape_y));

  make_empty_lcache(&cache);
  // forward

  ASSERT_EQ(S_OK, residual_basic_no_bn_forward(x, w1, w2, &cache, params, y));

#ifdef TEST_MORE
  double value_list[] = {
      -0.,        -0.,        -0.,        -0.,        -0.,        -0.,
      -0.,        -0.,        -0.,        -0.,        -0.,        -0.,
      -0.,        -0.,        -0.,        -0.,        -0.,        -0.,
      -0.,        -0.,        -0.,        -0.,        -0.,        -0.,
      -0.,        0.1036077,  0.17026859, 0.20490945, 0.19533295, 0.13077962,
      0.20268823, 0.3157693,  0.37083051, 0.34874428, 0.23300181, 0.29899355,
      0.45501809, 0.52511674, 0.48791309, 0.3254784,  0.32594793, 0.4836702,
      0.5509838,  0.50980631, 0.3452554,  0.24320672, 0.33982671, 0.38128489,
      0.3569817,  0.25830204, 0.39611021, 0.58334737, 0.66730938, 0.62644689,
      0.4413173,  0.64747984, 0.97797659, 1.12374872, 1.04684988, 0.71369172,
      0.86575658, 1.31510882, 1.505771,   1.39299934, 0.93723694, 0.9070172,
      1.36682036, 1.55789765, 1.43814524, 0.97151342, 0.67198689, 0.97384836,
      1.09768047, 1.01832896, 0.71440782, -0.,        -0.,        -0.,
      -0.,        -0.,        -0.,        -0.,        -0.,        -0.,
      -0.,        -0.,        -0.,        -0.,        -0.,        -0.,
      -0.,        -0.,        -0.,        -0.,        -0.,        -0.,
      -0.,        -0.,        -0.,        -0.,        1.09831069, 1.54835387,
      1.70389338, 1.54827522, 1.06948005, 1.51668713, 2.18534344, 2.41442392,
      2.17800891, 1.45756926, 1.70079805, 2.45917916, 2.71774033, 2.44753214,
      1.62811066, 1.56436813, 2.23379079, 2.46157262, 2.22226961, 1.49816805,
      1.0725285,  1.45479892, 1.58569319, 1.44958604, 1.03566807, 2.43506638,
      3.6308706,  4.04184763, 3.64389839, 2.41934218, 3.62782057, 5.50889042,
      6.15233568, 5.52105841, 3.58820522, 4.12460245, 6.28294107, 7.01828398,
      6.29048228, 4.07110329, 3.77567282, 5.71614398, 6.37538926, 5.7192327,
      3.72408263, 2.50945543, 3.69612185, 4.09955233, 3.6991228,  2.48080883};

  tensor_fill_list(y_ref, value_list, dim_of_shape(value_list));

  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
  if (tensor_rel_error(y_ref, y) > 1e-7) {
    tensor_dump(y);
    tensor_dump(y_ref);
  }
#endif

  // backward
  tensor_t dy = tensor_make_linspace_alike(0.1, 0.5, y);  // make it radom


  // output for backward
  tensor_t dx = tensor_make_zeros_alike(x);
  tensor_t dw1 = tensor_make_zeros_alike(w1);
  tensor_t dw2 = tensor_make_zeros_alike(w2);

  std::cout << "cache size " << cache.count << '\n';
  tensor_print_flat(cache.all_tensors[cache.count -1]);
  tensor_print_flat(cache.all_tensors[cache.count -2]);

  EXPECT_EQ(S_OK,
            residual_basic_no_bn_backward(dx, dw1, dw2, &cache, params, dy));

//  tensor_print_flat(x);
//  tensor_print_flat(dx);
//
//  tensor_print_flat(w1);
//  tensor_print_flat(dw1);
//
//  tensor_print_flat(w2);
//  tensor_print_flat(dw2);

  // numerical check
  tensor_t dx_ref = tensor_make_alike(x);
  tensor_t dw1_ref = tensor_make_alike(dw1);
  tensor_t dw2_ref = tensor_make_alike(dw2);

  // evaluate gradient of x
  eval_numerical_gradient(
      [w1, w2, params](tensor_t const in, tensor_t out) {
        residual_basic_no_bn_forward(in, w1, w2, NULL, params, out);
      },
      x, dy, dx_ref);
  ASSERT_LT(tensor_rel_error(dx_ref, dx), 1e-4);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w1
  eval_numerical_gradient(
      [x, w2, params](tensor_t const in, tensor_t out) {
        residual_basic_no_bn_forward(x, in, w2, NULL, params, out);
      },
      w1, dy, dw1_ref);
  ASSERT_LT(tensor_rel_error(dw1_ref, dw1), 1e-3);
  PINF("gradient check of w1... is ok");

  // evaluate gradient of w2
  eval_numerical_gradient(
      [x, w1, params](tensor_t const in, tensor_t out) {
        residual_basic_no_bn_forward(x, w1, in, NULL, params, out);
      },
      w2, dy, dw2_ref);
  ASSERT_LT(tensor_rel_error(dw2_ref, dw2), 1e-4);
  PINF("gradient check of w2... is ok");
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
