/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/layer_sandwich.h"
#include "awnndevice/tensor.cuh"
#include "gtest/gtest.h"
#include "test_util.h"

#include "awnndevice/layer_sandwich_device.cuh"
#define TEST_MORE

namespace {

// The fixture for testing class Foo.
class LayerResBlockDevice : public ::testing::Test {
 protected:
  LayerResBlockDevice() {
    // You can do set-up work for each test here.
    stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      PERR("CUBLAS initialization failed\n");
    }
  }

  ~LayerResBlockDevice() override {
    // You can do clean-up work that doesn't throw exceptions here.
    cublasDestroy(handle_);
  }

  // Objects declared here can be used by all tests in the test case for Foo.
  cublasHandle_t handle_;
  cublasStatus_t stat;
};

TEST_F(LayerResBlockDevice, ConvRelu) {
  lcache_t cache;

#ifdef TEST_MORE
  int shape_x[] = {2, 3, 8, 8};
  int shape_w[] = {4, 3, 3, 3};
  int shape_y[] = {2, 4, 8, 8};
#else
  int shape_x[] = {1, 1, 4, 4};
  int shape_w[] = {1, 1, 3, 3};
  int shape_y[] = {1, 1, 4, 4};
#endif

  conv_param_t params;
  params.stride = 1;
  params.padding = 1;

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, array_size(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, array_size(shape_w));

  tensor_t y = tensor_make_zeros(shape_y, array_size(shape_y));

  make_empty_lcache(&cache);

  tensor_t d_x = tensor_make_copy_h2d(x);
  tensor_t d_w = tensor_make_copy_h2d(w);
  tensor_t d_y = tensor_make_copy_h2d(y);
  // forward
  struct layer_context_device context = {
      .d_tmp = tensor_make_alike_device(d_y),
      .d_dtmp = tensor_make_alike_device(d_y),
  };
  EXPECT_EQ(S_OK, conv_relu_forward_device(handle_, d_x, d_w, &cache, params,
                                           d_y, &context));

  // backward
  PINF("Now Backward");
  tensor_t dy = tensor_make_linspace_alike(0.1, 0.5, y);  // make it radom

  // output for backward
  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  tensor_t d_dy = tensor_make_copy_h2d(dy);
  tensor_t d_dx = tensor_make_copy_h2d(dx);
  tensor_t d_dw = tensor_make_copy_h2d(dw);

  EXPECT_EQ(S_OK, conv_relu_backward_device(handle_, d_dx, d_dw, &cache, params,
                                            d_dy, &context));

  tensor_copy_d2h(dx, d_dx);
  tensor_copy_d2h(dw, d_dw);

  // numerical check
  tensor_t dx_ref = tensor_make_alike(x);
  tensor_t dw_ref = tensor_make_alike(w);

  // evaluate gradient of x
  eval_numerical_gradient(
      [w, params](tensor_t const in, tensor_t out) {
        conv_relu_forward(in, w, NULL, params, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-3);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [x, params](tensor_t const in, tensor_t out) {
        conv_relu_forward(x, in, NULL, params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-3);
  PINF("gradient check of w... is ok");

  layer_context_destroy_device(&context);
}

TEST_F(LayerResBlockDevice, ResidualBlock) {
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

  // copy input to device
  tensor_t d_x = tensor_make_copy_h2d(x);
  tensor_t d_w1 = tensor_make_copy_h2d(w1);
  tensor_t d_w2 = tensor_make_copy_h2d(w2);
  tensor_t d_y = tensor_make_copy_h2d(y);

  struct layer_context_device *context;
  resblock_create_context_device(&context, d_y);  // 1+2 contexts

  ASSERT_EQ(S_OK, resblock_forward_device(handle_, d_x, d_w1, d_w2, &cache,
                                          params, d_y, context));

  // copy output back
  tensor_copy_d2h(y, d_y);

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

  ASSERT_LT(tensor_rel_error(y_ref, y), 1e-3);
  if (tensor_rel_error(y_ref, y) > 1e-3) {
    tensor_dump(y);
    tensor_dump(y_ref);
  } else {
    PINF("forward value checked!!!!!");
  }
#endif

  // backward
  tensor_t dy = tensor_make_linspace_alike(0.1, 0.5, y);  // make it radom

  // output for backward
  tensor_t dx = tensor_make_alike(x);
  tensor_t dw1 = tensor_make_alike(w1);
  tensor_t dw2 = tensor_make_alike(w2);

  // copy input to device mem
  tensor_t d_dx = tensor_make_copy_h2d(dx);
  tensor_t d_dw1 = tensor_make_copy_h2d(dw1);
  tensor_t d_dw2 = tensor_make_copy_h2d(dw2);
  tensor_t d_dy = tensor_make_copy_h2d(dy);

  EXPECT_EQ(S_OK, resblock_backward_device(handle_, d_dx, d_dw1, d_dw2, &cache,
                                           params, d_dy, context));

  // copy gradient back
  tensor_copy_d2h(dx, d_dx);
  tensor_copy_d2h(dw1, d_dw1);
  tensor_copy_d2h(dw2, d_dw2);

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
  ASSERT_LT(tensor_rel_error(dx_ref, dx), 1e-3);
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
  ASSERT_LT(tensor_rel_error(dw2_ref, dw2), 1e-3);
  PINF("gradient check of w2... is ok");

  tensor_destroy_device(&d_x);
  tensor_destroy_device(&d_w1);
  tensor_destroy_device(&d_w2);
  tensor_destroy_device(&d_y);
  tensor_destroy_device(&d_dx);
  tensor_destroy_device(&d_dw1);
  tensor_destroy_device(&d_dw2);
  tensor_destroy_device(&d_dy);
  resblock_destroy_context_device(context);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
