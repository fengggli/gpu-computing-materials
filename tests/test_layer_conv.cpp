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
  static tensor_t x, w;
  static uint nr_img;
  static uint sz_img; // width and height of filter
  static uint nr_in_channel;
  static uint sz_filter; // width and height of filter
  static uint nr_filter; // # of output channel

  static lcache_t cache;
  static conv_param_t params;
};

//
tensor_t LayerConvTest::x;
tensor_t LayerConvTest::w;

uint LayerConvTest::nr_img;
uint LayerConvTest::sz_img;
uint LayerConvTest::nr_in_channel;
uint LayerConvTest::sz_filter;
uint LayerConvTest::nr_filter;

lcache_t LayerConvTest::cache;
conv_param_t LayerConvTest::params;

TEST_F(LayerConvTest, Construct) {
  params.stride=2;
  params.padding=1;

  nr_img = 2;
  sz_img = 4;
  nr_in_channel = 3;
  sz_filter = 4;
  nr_filter =3;

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img}; // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter, sz_filter}; // 3x3x4x4

  x   = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  w   = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));


  make_empty_lcache(&cache);
}

TEST_F(LayerConvTest, Forward){

  uint sz_out = 1 + (sz_img + 2*params.padding - sz_filter)/params.stride;
  EXPECT_EQ(2, sz_out);
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out}; // 2x3x2x2

  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  status_t ret = convolution_forward(x, w, &cache, params, y);// foward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  tensor_t y_ref = tensor_make_alike(y);
  T value_list[] = {0.02553947, 0.01900658, -0.03984868, -0.09432237,
                    0.05964474,  0.09894079, 0.12641447,  0.19823684,
                    0.09375, 0.178875,0.29267763,0.49079605,
                    -0.36098684 -0.57783553,-0.67079605 -1.06632237,
                    0.28701316, 0.42294079, 0.41630921,  0.6075,
                    0.93501316,  1.42371711, 1.50341447,  2.28132237};

  tensor_fill_list(y_ref, value_list, dim_of_shape(value_list));

  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
  PINF("Consistent with expected results");

}

TEST_F(LayerConvTest, DISABLED_Backward){
  status_t ret;

  uint sz_out = 1 + (sz_img + 2*params.padding - sz_filter)/params.stride;
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out}; // 4x2x5x5

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));

  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  ret = convolution_backward(dx, dw, &cache, dy); // backward needs to call free_lcache(cache);
  EXPECT_EQ(ret, S_OK);

  /* II. Numerical check */
  // I had to make this copy since lambda doesn't allow me to use global
  // variable
  tensor_t x_copy = tensor_make_copy(x);
  tensor_t w_copy = tensor_make_copy(w);

  tensor_t dx_ref = tensor_make_alike(x);
  tensor_t dw_ref = tensor_make_alike(w);

  // evaluate gradient of x
  eval_numerical_gradient(
      [w_copy](tensor_t const in, tensor_t out) {
        convolution_forward(in, w_copy, NULL, params, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [x_copy](tensor_t const in, tensor_t out) {
        convolution_forward(x_copy, in , NULL, params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-7);
  PINF("gradient check of w... is ok");

  EXPECT_EQ(ret, S_OK);
}

// TODO: check with cudnn

TEST_F(LayerConvTest,CheckLcache){
  EXPECT_EQ(cache.count, 0); // backward needs to call free_lcache(cache);
}



TEST_F(LayerConvTest, Destroy) {
  tensor_destroy(x);
  tensor_destroy(w);
  free_lcache(&cache);
}


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
