/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

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

  x= tensor_make(shape_x, dim_of_shape(shape_x));
  dx= tensor_make(shape_x, dim_of_shape(shape_x));
  y= tensor_make(shape_y, dim_of_shape(shape_y));
  dy= tensor_make(shape_y, dim_of_shape(shape_y));

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
TEST_F(LayerGlobalAvgPoolTest, DISABLED_Forward){
  status_t ret;
  ret = global_avg_pool_forward(x, &cache, y);// foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, DISABLED_Backward){
  status_t ret;
  ret = global_avg_pool_backward(dx, &cache, dy); // backward needs to call free_lcache(cache);
  EXPECT_EQ(ret, S_OK);
}

// TODO: check with cudnn
// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest,CheckLcache){
  EXPECT_EQ(cache.count, 0); // backward needs to call free_lcache(cache);
}

// TODO : document tests
TEST_F(LayerGlobalAvgPoolTest, Destroy) {
  tensor_destroy(x);
  tensor_destroy(dx);
  tensor_destroy(y);
  tensor_destroy(dy);
  free_lcache(&cache);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
