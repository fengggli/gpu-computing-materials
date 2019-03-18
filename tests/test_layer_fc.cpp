/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "test_util.h"
#include "awnn/layer_fc.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"

namespace {

// The fixture for testing class Foo.
class LayerFcTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  LayerFcTest() {
    // You can do set-up work for each test here.
  }

  ~LayerFcTest() override {
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
  static tensor_t x, dx, w, dw, b, db, y, dy;
  static lcache_t cache;
};

//
tensor_t LayerFcTest::x;
tensor_t LayerFcTest::dx;
tensor_t LayerFcTest::w;
tensor_t LayerFcTest::dw;
tensor_t LayerFcTest::b;
tensor_t LayerFcTest::db;
tensor_t LayerFcTest::y;
tensor_t LayerFcTest::dy;
lcache_t LayerFcTest::cache;

TEST_F(LayerFcTest, Construct) {
  uint const shape_x[] = {2, 4, 5, 6};
  uint const shape_w[] = {120, 3};
  uint const shape_b[] = {3};
  uint const shape_y[] = {2, 3}; // (7-3+2*padding)/stride +1 = 7
  x   = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  dx  = tensor_make(shape_x, dim_of_shape(shape_x));
  w   = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  dw  = tensor_make(shape_w, dim_of_shape(shape_w));
  b   = tensor_make_linspace(-0.3, 0.1,shape_b, dim_of_shape(shape_b));
  db  = tensor_make(shape_b, dim_of_shape(shape_b));
  y   = tensor_make(shape_y, dim_of_shape(shape_y));
  dy  = tensor_make(shape_y, dim_of_shape(shape_y));

  make_empty_lcache(&cache);
}

TEST_F(LayerFcTest, Forward){
  status_t ret;
  ret = layer_fc_forward(x,w,b, &cache,y);// forward function should allocate and populate cache;
  tensor_dump(y);
  EXPECT_EQ(ret, S_OK);
}

TEST_F(LayerFcTest, Backward) {
  status_t ret;
  ret = layer_fc_backward(dx, dw, db, &cache, dy); // backward needs to call free_lcache(cache);
  EXPECT_EQ(ret, S_OK);
}

// TODO: check with cudnn

TEST_F(LayerFcTest,CheckLcache){
  EXPECT_EQ(cache.count, 0); // backward needs to call free_lcache(cache);
}




TEST_F(LayerFcTest, Destroy) {
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
