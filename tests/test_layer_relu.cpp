/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/layer_relu.h"
#include "awnn/tensor.h"
#include "test_util.h"
#include "gtest/gtest.h"

namespace {

// The fixture for testing class Foo.
class LayerReluTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  LayerReluTest() {
    // You can do set-up work for each test here.
  }

  ~LayerReluTest() override {
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

  // Objects declared here can be used by all tests.
  static tensor_t x; // they have to be global since I also use them for
                     // numerical gradient check
  static lcache_t cache;
};

// input of forward
tensor_t LayerReluTest::x;

lcache_t LayerReluTest::cache;

TEST_F(LayerReluTest, Construct) {

  uint const shape_x[] = {3, 4}; // e.g. 2 images, 4 channels, width =5 hight=6
  x = tensor_make_linspace(-0.5, 0.5, shape_x, dim_of_shape(shape_x));

  make_empty_lcache(&cache);
}

TEST_F(LayerReluTest, Forward) {
  /* I. Perform the forwarding */
  status_t ret;
  tensor_t y = tensor_make_alike(x);

  ret = layer_relu_forward(
      x, &cache, y); // forward function should allocate and populate cache;
  PINF("calculated y:");
  tensor_dump(y);
  EXPECT_EQ(ret, S_OK);

  /* II. Check with expected results */
  tensor_t y_ref = tensor_make_alike(y);
  // values from fc assignment in cs231n
  double value_list[] = {
      0,          0,          0,          0,          0.,         0.,
      0.04545455, 0.13636364, 0.22727273, 0.31818182, 0.40909091, 0.5,
  };
  tensor_fill_list(y_ref, value_list, dim_of_shape(value_list));

  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
  PINF("Consistent with expected results");

  tensor_destroy(&y_ref);
}

TEST_F(LayerReluTest, Backward) {
  /* I. Perform the backwarding */
  status_t ret;

  // input for backward
  tensor_t dy = tensor_make_linspace_alike(0.1, 0.5, x); // make it radom

  // output for backward
  tensor_t dx = tensor_make_alike(x);

  ret = layer_relu_backward(dx, &cache, dy);
  EXPECT_EQ(ret, S_OK);

  /* II. Numerical check */

  tensor_t dx_ref = tensor_make_alike(x);
  // tensor_dump(dx);

  // evaluate gradient of x
  eval_numerical_gradient(
      [](tensor_t const in, tensor_t out) {
        layer_relu_forward(in, NULL, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
  PINF("gradient check of x... is ok");

}

TEST_F(LayerReluTest, CheckLcache) {
  EXPECT_EQ(cache.count, 0); // backward needs to call lcache_free_all(cache);
  }

  TEST_F(LayerReluTest, Destroy) {
    tensor_destroy(&x);

    lcache_free_all(&cache);
  }


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
