/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/channel.h"
#include "test_util.h"
#include "awnn/layer_pool.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
namespace {

// The fixture for testing class Foo.
class LayerGlobalAvgPoolTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  LayerGlobalAvgPoolTest() {
    // You can do set-up work for each test here.
  }

  ~LayerGlobalAvgPoolTest() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

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
  uint const shape_x[] = {6, 2, 7, 7}; //6 images, 2x7x7
  uint const shape_y[] = {6, 2, 1, 1};

  x = tensor_make(shape_x, dim_of_shape(shape_x));
  dx = tensor_make(shape_x, dim_of_shape(shape_x));
  y = tensor_make(shape_y, dim_of_shape(shape_y));
  dy = tensor_make(shape_y, dim_of_shape(shape_y));

  make_empty_lcache(&cache);
}

// channel_mean
TEST_F(LayerGlobalAvgPoolTest, channel_mean){
  uint channel_capacity = x.dim.dims[2] * x.dim.dims[3]; // each chanel
  uint target_channel = 0;
  T* start = x.data + channel_capacity * target_channel;

  T res = channel_mean(start, channel_capacity);
}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, Forward){
  uint const shape_x[] = {6, 2, 7, 7}; //6 images, 2x7x7
  uint const shape_y[] = {6, 2, 1, 1};

  tensor_t in = tensor_make(shape_x, dim_of_shape(shape_x));
  tensor_t din = tensor_make(shape_x, dim_of_shape(shape_x));
  tensor_t out = tensor_make(shape_y, dim_of_shape(shape_y));
  tensor_t dout = tensor_make(shape_y, dim_of_shape(shape_y));
  
  status_t ret;

  ret = global_avg_pool_forward(in, NULL, out);// foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, Backward){
  uint const shape_x[] = {6, 2, 7, 7}; //6 images, 2x7x7
  uint const shape_y[] = {6, 2, 1, 1};

  tensor_t x = tensor_make_linspace(0.1,0.3, shape_x, dim_of_shape(shape_x));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  status_t ret;

  ret = global_avg_pool_forward(x, &cache, y);// foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);


  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5,shape_y, dim_of_shape(shape_y)); // some fake data

  // output for backward
  tensor_t dx = tensor_make_alike(x);

  ret = global_avg_pool_backward(dx, &cache, dy); // backward needs to call lcache_free_all(cache);
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

// TODO: check with cudnn
// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, CheckLcache){
  EXPECT_EQ(cache.count, 0); // backward needs to call lcache_free_all(cache);
}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, Destroy) {
  tensor_destroy(x);
  tensor_destroy(dx);
  tensor_destroy(y);
  tensor_destroy(dy);
  lcache_free_all(&cache);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
