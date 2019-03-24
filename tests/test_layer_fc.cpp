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

  // Objects declared here can be used by all tests.
  static tensor_t x, w, b; // they have to be global since I also use them for
                           // numerical gradient check
  static lcache_t cache;
};

// input of forward
tensor_t LayerFcTest::x;
tensor_t LayerFcTest::w;
tensor_t LayerFcTest::b;

lcache_t LayerFcTest::cache;

TEST_F(LayerFcTest, Construct) {

  uint const shape_x[] = {2, 4, 5,
                          6}; // e.g. 2 images, 4 channels, width =5 hight=6
  uint const shape_w[] = {120, 3}; // fc 120~3  neurons
  uint const shape_b[] = {3};      // 3n
  x   = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  w   = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  b   = tensor_make_linspace(-0.3, 0.1,shape_b, dim_of_shape(shape_b));

  make_empty_lcache(&cache);
}

TEST_F(LayerFcTest, Forward){
  /* I. Perform the forwarding */
  status_t ret;
  uint const shape_y[] = {x.dim.dims[0], w.dim.dims[1]};
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  ret = layer_fc_forward(x,w,b, &cache,y);// forward function should allocate and populate cache;
  PINF("calculated y:");
  tensor_dump(y);
  EXPECT_EQ(ret, S_OK);

  /* II. Check with expected results */
  tensor_t y_ref = tensor_make_alike(y);
  // values from fc assignment in cs231n
  T value_list[] = {1.49834967, 1.70660132, 1.91485297,
                    3.25553199, 3.5141327,  3.77273342};
  tensor_fill_list(y_ref, value_list, dim_of_shape(value_list));

  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
  PINF("Consistent with expected results");

  tensor_destroy(y_ref);
}

TEST_F(LayerFcTest, Backward) {
  /* I. Perform the backwarding */
  status_t ret;
  uint const shape_y[] = {x.dim.dims[0], w.dim.dims[1]};

  // input for backward
  tensor_t dy = tensor_make_linspace(0.1, 0.5, shape_y,
                                     dim_of_shape(shape_y)); // make it radom

  // output for backward
  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);
  tensor_t db = tensor_make_alike(b);

  ret = layer_fc_backward(dx, dw, db, &cache, dy);
  EXPECT_EQ(ret, S_OK);

  /* II. Numerical check */
  // I had to make this copy since lambda doesn't allow me to use global
  // variable
  tensor_t x_copy = tensor_make_copy(x);
  tensor_t w_copy = tensor_make_copy(w);
  tensor_t b_copy = tensor_make_copy(b);

  tensor_t dx_ref = tensor_make_alike(x);
  tensor_t dw_ref = tensor_make_alike(w);
  tensor_t db_ref = tensor_make_alike(b);
  // tensor_dump(dx);

  // evaluate gradient of x
  eval_numerical_gradient(
      [w_copy, b_copy](tensor_t const in, tensor_t out) {
        layer_fc_forward(in, w_copy, b_copy, NULL, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [x_copy, b_copy](tensor_t const in, tensor_t out) {
        layer_fc_forward(x_copy, in, b_copy, NULL, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-7);
  PINF("gradient check of w... is ok");

  // evaluate gradient of b
  eval_numerical_gradient(
      [x_copy, w_copy](tensor_t const in, tensor_t out) {
        layer_fc_forward(x_copy, w_copy, in, NULL, out);
      },
      b, dy, db_ref);
  EXPECT_LT(tensor_rel_error(db_ref, db), 1e-7);
  PINF("gradient check of b... is ok");
}

TEST_F(LayerFcTest, CheckLcache) {
  EXPECT_EQ(cache.count, 0); // backward needs to call free_lcache(cache);
  }

  TEST_F(LayerFcTest, Destroy) {
    tensor_destroy(x);
    tensor_destroy(w);
    tensor_destroy(b);

    free_lcache(&cache);
  }


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
