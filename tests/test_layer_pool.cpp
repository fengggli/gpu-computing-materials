/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/channel.h"
#include "awnn/layer_pool.h"
#include "awnn/tensor.h"
#include "config.h"
#include "gtest/gtest.h"
#include "test_util.h"
namespace {

// The fixture for testing class Foo.
class LayerGlobalAvgPoolTest : public ::testing::Test {
 protected:
  // Objects declared here can be used by all tests in the test case for Foo.
  static tensor_t x, dx, y, dy;
  static lcache_t cache;
};

//
tensor_t LayerGlobalAvgPoolTest::x;
tensor_t LayerGlobalAvgPoolTest::dx;
tensor_t LayerGlobalAvgPoolTest::y;
tensor_t LayerGlobalAvgPoolTest::dy;
lcache_t LayerGlobalAvgPoolTest::cache;

// skeleton
TEST_F(LayerGlobalAvgPoolTest, Construct) {
  uint const shape_x[] = {6, 2, 7, 7};  // 6 images, 2x7x7
  uint const shape_y[] = {6, 2, 1, 1};

  x = tensor_make(shape_x, dim_of_shape(shape_x));
  dx = tensor_make(shape_x, dim_of_shape(shape_x));
  y = tensor_make(shape_y, dim_of_shape(shape_y));
  dy = tensor_make(shape_y, dim_of_shape(shape_y));

  make_empty_lcache(&cache);
}

// channel_mean
TEST_F(LayerGlobalAvgPoolTest, channel_mean) {
  uint channel_capacity = x.dim.dims[2] * x.dim.dims[3];  // each chanel
  uint target_channel = 0;
  T* start = x.data + channel_capacity * target_channel;

  T res = channel_mean(start, channel_capacity);
  PINF("channel mean, %.3f", res);
}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, Forward) {
  uint const shape_x[] = {6, 2, 7, 7};  // 6 images, 2x7x7
  uint const shape_y[] = {6, 2, 1, 1};

  tensor_t in = tensor_make(shape_x, dim_of_shape(shape_x));
  tensor_t out = tensor_make(shape_y, dim_of_shape(shape_y));

  status_t ret;

  ret = global_avg_pool_forward(
      in, NULL, out);  // foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);
}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, Backward) {
  uint const shape_x[] = {6, 2, 7, 7};  // 6 images, 2x7x7
  uint const shape_y[] = {6, 2, 1, 1};

  tensor_t x = tensor_make_linspace(0.1, 0.3, shape_x, dim_of_shape(shape_x));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  status_t ret;

  ret = global_avg_pool_forward(
      x, &cache, y);  // foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  // check forward value
  tensor_t y_ref = tensor_make_alike(y);
  double value_list[] = {0.10817717, 0.12487223, 0.14156729, 0.15826235,
                         0.17495741, 0.19165247, 0.20834753, 0.22504259,
                    0.24173765, 0.25843271, 0.27512777, 0.29182283};
  tensor_fill_list(y_ref, value_list, dim_of_shape(value_list));
  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y,
                                     dim_of_shape(shape_y));  // some fake data

  // output for backward
  tensor_t dx = tensor_make_alike(x);

  ret = global_avg_pool_backward(
      dx, &cache, dy);  // backward needs to call lcache_free_all(cache);
  EXPECT_EQ(ret, S_OK);

  /* II. Numerical check */
  // variable
  tensor_t dx_ref = tensor_make_alike(x);

  // evaluate gradient of x
  eval_numerical_gradient(
      [](tensor_t const in, tensor_t out) {
        global_avg_pool_forward(in, NULL, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 3e-3);
  PINF("gradient check of x... is ok");
}

#ifdef USE_CUDA
TEST_F(LayerGlobalAvgPoolTest, BackwardDevice) {
  uint const shape_x[] = {6, 2, 7, 7};  // 6 images, 2x7x7
  uint const shape_y[] = {6, 2, 1, 1};

  tensor_t x = tensor_make_linspace(0.1, 0.3, shape_x, dim_of_shape(shape_x));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  status_t ret;

  ret = global_avg_pool_forward_device(
      x, &cache, y);  // foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  // check forward value
  tensor_t y_ref = tensor_make_alike(y);
  double value_list[] = {0.10817717, 0.12487223, 0.14156729, 0.15826235,
                         0.17495741, 0.19165247, 0.20834753, 0.22504259,
                         0.24173765, 0.25843271, 0.27512777, 0.29182283};
  tensor_fill_list(y_ref, value_list, dim_of_shape(value_list));
  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y,
                                     dim_of_shape(shape_y));  // some fake data

  // output for backward
  tensor_t dx = tensor_make_alike(x);

  ret = global_avg_pool_backward_device(
      dx, &cache, dy);  // backward needs to call lcache_free_all(cache);
  EXPECT_EQ(ret, S_OK);

  /* II. Numerical check */
  // variable
  tensor_t dx_ref = tensor_make_alike(x);

  // evaluate gradient of x
  eval_numerical_gradient(
      [](tensor_t const in, tensor_t out) {
        global_avg_pool_forward(in, NULL, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
  PINF("gradient check of x... is ok");
}
#endif

// TODO : check with cudnn
// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, CheckLcache) {
  EXPECT_EQ(cache.count, 0);  // backward needs to call lcache_free_all(cache);
}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, Destroy) {
  tensor_destroy(&x);
  tensor_destroy(&dx);
  tensor_destroy(&y);
  tensor_destroy(&dy);
  lcache_free_all(&cache);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
