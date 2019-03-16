/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "test_util.h"
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"

namespace {

// The fixture for testing class Foo.
class LayerConvTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  LayerConvTest() {
    // You can do set-up work for each test here.
  }

  ~LayerConvTest() override {
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
  static tensor_t x, w, dx, dw, y, dy;
  static lcache_t cache;
  static conv_param_t params;
};

//
tensor_t LayerConvTest::x;
tensor_t LayerConvTest::dx;
tensor_t LayerConvTest::w;
tensor_t LayerConvTest::dw;
tensor_t LayerConvTest::y;
tensor_t LayerConvTest::dy;
lcache_t LayerConvTest::cache;
conv_param_t LayerConvTest::params;

TEST_F(LayerConvTest, Construct) {
  params.stride=1;
  params.padding=1;
  uint const shape_x[] = {6, 2, 7, 7};
  uint const shape_w[] = {4, 2, 3, 3};
  uint const shape_y[] = {6, 4, 7, 7}; // (7-3+2*padding)/stride +1 = 7
  x   = tensor_make(shape_x, dim_of_shape(shape_x));
  dx  = tensor_make(shape_x, dim_of_shape(shape_x));
  w   = tensor_make(shape_w, dim_of_shape(shape_w));
  dw  = tensor_make(shape_w, dim_of_shape(shape_w));
  y   = tensor_make(shape_y, dim_of_shape(shape_y));
  dy  = tensor_make(shape_y, dim_of_shape(shape_y));

  make_empty_lcache(&cache);
}

TEST_F(LayerConvTest, DISABLED_Forward){
  status_t ret;
  ret = convolution_forward(x,w, &cache, params,y);// foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

}

TEST_F(LayerConvTest, DISABLED_Backward){
  status_t ret;
  ret = convolution_backward(dx, dw, &cache, dy); // backward needs to call free_lcache(cache);
  EXPECT_EQ(ret, S_OK);
}

// TODO: check with cudnn

TEST_F(LayerConvTest,CheckLcache){
  EXPECT_EQ(cache.count, 0); // backward needs to call free_lcache(cache);
}




TEST_F(LayerConvTest, Destroy) {
  tensor_destroy(x);
  tensor_destroy(dx);
  tensor_destroy(w);
  tensor_destroy(dw);
  tensor_destroy(y);
  tensor_destroy(dy);
  free_lcache(&cache);
}


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
