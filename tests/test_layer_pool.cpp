/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

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

// Tests that the Foo::Bar() method does Abc.
TEST_F(LayerGlobalAvgPoolTest, Construct) {
  uint const shape_x[] = {6, 2, 7, 7};
  uint const shape_y[] = {6, 4, 1, 1}; // (7-3+2*padding)/stride +1 = 7
  x= tensor_make(shape_x, 3);
  dx= tensor_make(shape_x, 3);
  y= tensor_make(shape_y, 3);
  dy= tensor_make(shape_y, 3);

  make_empty_lcache(&cache);
}

TEST_F(LayerGlobalAvgPoolTest, DISABLED_Forward){
  status_t ret;
  ret = global_avg_pool_forward(x,&cache,y);// foward function should allocate and populate cache;
  EXPECT_TRUE(ret == S_OK);

}

TEST_F(LayerGlobalAvgPoolTest, DISABLED_Backward){
  status_t ret;
  ret = global_avg_pool_backward(dx, &cache, dy); // backward needs to call free_lcache(cache);
  EXPECT_TRUE(ret == S_OK);
}

// TODO: check with cudnn

TEST_F(LayerGlobalAvgPoolTest,CheckLcache){
  EXPECT_TRUE(cache.count == 0); // backward needs to call free_lcache(cache);
}

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