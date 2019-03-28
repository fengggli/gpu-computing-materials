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
//  static tensor_t x, w;
//  static uint nr_img;
//  static uint sz_img; // width and height of filter
//  static uint nr_in_channel;
//  static uint sz_filter; // width and height of filter
//  static uint nr_filter; // # of output channel
//
//  static lcache_t cache;
//  static conv_param_t params;
};

//tensor_t LayerConvTest::x;
//tensor_t LayerConvTest::w;
//
//uint LayerConvTest::nr_img;
//uint LayerConvTest::sz_img;
//uint LayerConvTest::nr_in_channel;
//uint LayerConvTest::sz_filter;
//uint LayerConvTest::nr_filter;
//
//lcache_t LayerConvTest::cache;
//conv_param_t LayerConvTest::params;


TEST_F(LayerConvTest, im2col_numerical1) {
  conv_param_t conv_params = {2, 1};

  uint num_img = 2;
  uint img_sz = 4;
  uint nr_in_channel = 3;
  uint fil_sz = 4;
  uint nr_filter =3;
  uint sz_out = 1 + (img_sz + 2 * conv_params.padding - fil_sz) / conv_params.stride;

  uint const shape_x[] = {num_img, nr_in_channel, img_sz, img_sz}; // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, fil_sz, fil_sz}; // 3x3x4x4

  EXPECT_EQ(2, sz_out);

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));

  lcache_t cache;
  make_empty_lcache(&cache);

  tensor_t ret = im2col(x, w, conv_params);// forward function should allocate and populate cache;
  uint const shape_ref[] = {384};
  tensor_reshape_(&ret, shape_ref, dim_of_shape(shape_ref));

  tensor_t ref = tensor_make(shape_ref, dim_of_shape(shape_ref));
  T value_list[] =
    { 0.               ,  0.               ,  0.               ,
      0.               ,  0.               ,  0.               ,
      -0.068421052631579,  0.234736842105263,  0.               ,
      0.               ,  0.               ,  0.               ,
      -0.074736842105263,  0.228421052631579, -0.062105263157895,
      0.241052631578947,  0.               ,  0.               ,
      0.               ,  0.               , -0.068421052631579,
      0.234736842105263, -0.055789473684211,  0.247368421052632,
      0.               ,  0.               ,  0.               ,
      0.               , -0.062105263157895,  0.241052631578947,
      0.               ,  0.               ,  0.               ,
      0.               , -0.093684210526316,  0.209473684210526,
      0.               ,  0.               , -0.043157894736842,
      0.26             , -0.1              ,  0.203157894736842,
      -0.087368421052632,  0.215789473684211, -0.049473684210526,
      0.253684210526316, -0.036842105263158,  0.266315789473684,
      -0.093684210526316,  0.209473684210526, -0.081052631578947,
      0.222105263157895, -0.043157894736842,  0.26             ,
      -0.030526315789474,  0.272631578947368, -0.087368421052632,
      0.215789473684211,  0.               ,  0.               ,
      -0.036842105263158,  0.266315789473684,  0.               ,
      0.               ,  0.               ,  0.               ,
      -0.068421052631579,  0.234736842105263,  0.               ,
      0.               , -0.017894736842105,  0.285263157894737,
      -0.074736842105263,  0.228421052631579, -0.062105263157895,
      0.241052631578947, -0.024210526315789,  0.278947368421053,
      -0.011578947368421,  0.291578947368421, -0.068421052631579,
      0.234736842105263, -0.055789473684211,  0.247368421052632,
      -0.017894736842105,  0.285263157894737, -0.005263157894737,
      0.297894736842105, -0.062105263157895,  0.241052631578947,
      0.               ,  0.               , -0.011578947368421,
      0.291578947368421,  0.               ,  0.               ,
      0.               ,  0.               , -0.043157894736842,
      0.26             ,  0.               ,  0.               ,
      0.               ,  0.               , -0.049473684210526,
      0.253684210526316, -0.036842105263158,  0.266315789473684,
      0.               ,  0.               ,  0.               ,
      0.               , -0.043157894736842,  0.26             ,
      -0.030526315789474,  0.272631578947368,  0.               ,
      0.               ,  0.               ,  0.               ,
      -0.036842105263158,  0.266315789473684,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.               ,  0.032631578947368,
      0.335789473684211,  0.               ,  0.               ,
      0.               ,  0.               ,  0.026315789473684,
      0.329473684210526,  0.038947368421053,  0.342105263157895,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.032631578947368,  0.335789473684211,
      0.045263157894737,  0.348421052631579,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.038947368421053,  0.342105263157895,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.007368421052632,  0.310526315789474,  0.               ,
      0.               ,  0.057894736842105,  0.361052631578947,
      0.001052631578947,  0.304210526315789,  0.013684210526316,
      0.316842105263158,  0.051578947368421,  0.354736842105263,
      0.064210526315789,  0.367368421052632,  0.007368421052632,
      0.310526315789474,  0.02             ,  0.323157894736842,
      0.057894736842105,  0.361052631578947,  0.070526315789474,
      0.373684210526316,  0.013684210526316,  0.316842105263158,
      0.               ,  0.               ,  0.064210526315789,
      0.367368421052632,  0.               ,  0.               ,
      0.               ,  0.               ,  0.032631578947368,
      0.335789473684211,  0.               ,  0.               ,
      0.083157894736842,  0.386315789473684,  0.026315789473684,
      0.329473684210526,  0.038947368421053,  0.342105263157895,
      0.076842105263158,  0.38             ,  0.089473684210526,
      0.392631578947368,  0.032631578947368,  0.335789473684211,
      0.045263157894737,  0.348421052631579,  0.083157894736842,
      0.386315789473684,  0.095789473684211,  0.398947368421053,
      0.038947368421053,  0.342105263157895,  0.               ,
      0.               ,  0.089473684210526,  0.392631578947368,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.057894736842105,  0.361052631578947,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.051578947368421,  0.354736842105263,
      0.064210526315789,  0.367368421052632,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.057894736842105,  0.361052631578947,  0.070526315789474,
      0.373684210526316,  0.               ,  0.               ,
      0.               ,  0.               ,  0.064210526315789,
      0.367368421052632,  0.               ,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.133684210526316,  0.436842105263158,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.127368421052632,  0.430526315789474,
      0.14             ,  0.443157894736842,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.133684210526316,  0.436842105263158,  0.146315789473684,
      0.449473684210526,  0.               ,  0.               ,
      0.               ,  0.               ,  0.14             ,
      0.443157894736842,  0.               ,  0.               ,
      0.               ,  0.               ,  0.108421052631579,
      0.411578947368421,  0.               ,  0.               ,
      0.158947368421053,  0.462105263157895,  0.102105263157895,
      0.405263157894737,  0.114736842105263,  0.417894736842105,
      0.152631578947368,  0.455789473684211,  0.165263157894737,
      0.468421052631579,  0.108421052631579,  0.411578947368421,
      0.121052631578947,  0.424210526315789,  0.158947368421053,
      0.462105263157895,  0.171578947368421,  0.474736842105263,
      0.114736842105263,  0.417894736842105,  0.               ,
      0.               ,  0.165263157894737,  0.468421052631579,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.133684210526316,  0.436842105263158,
      0.               ,  0.               ,  0.184210526315789,
      0.487368421052632,  0.127368421052632,  0.430526315789474,
      0.14             ,  0.443157894736842,  0.177894736842105,
      0.481052631578947,  0.190526315789474,  0.493684210526316,
      0.133684210526316,  0.436842105263158,  0.146315789473684,
      0.449473684210526,  0.184210526315789,  0.487368421052632,
      0.196842105263158,  0.5              ,  0.14             ,
      0.443157894736842,  0.               ,  0.               ,
      0.190526315789474,  0.493684210526316,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.158947368421053,  0.462105263157895,  0.               ,
      0.               ,  0.               ,  0.               ,
      0.152631578947368,  0.455789473684211,  0.165263157894737,
      0.468421052631579,  0.               ,  0.               ,
      0.               ,  0.               ,  0.158947368421053,
      0.462105263157895,  0.171578947368421,  0.474736842105263,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.165263157894737,  0.468421052631579,
      0.               ,  0.               ,  0.               ,
      0.               ,  0.               ,  0.               };

  tensor_fill_list(ref, value_list, array_size(value_list));

  EXPECT_LT(tensor_rel_error(ref, ret), 1e-7);
  PINF("Consistent with expected results");
}

TEST_F(LayerConvTest, im2col_numerical2) {
  conv_param_t conv_params = {1, 0};

  uint n = 1;
  uint img_sz = 3;
  uint c = 2;
  uint fltr_sz = 2;
  uint num_fil = 2;
  uint sz_out = 1 + (img_sz + 2 * conv_params.padding - fltr_sz) / conv_params.stride;

  uint const shape_x[] = {n, c, img_sz, img_sz}; // 2x3x4x4
  uint const shape_w[] = {num_fil, c, fltr_sz, fltr_sz}; // 3x3x4x4

  EXPECT_EQ(2, sz_out);

  T x_values[] = { 1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2 };
  tensor_t x = tensor_make(shape_x, dim_of_shape(shape_x));
  tensor_fill_list(x, x_values, array_size(x_values));

  tensor_t w = tensor_make(shape_w, dim_of_shape(shape_w));

  lcache_t cache;
  make_empty_lcache(&cache);

  tensor_t ret = im2col(x, w, conv_params);// forward function should allocate and populate cache;

  uint const shape_ref[] = {32};
  tensor_reshape_(&ret, shape_ref, dim_of_shape(shape_ref));

  tensor_t ref = tensor_make(shape_ref, dim_of_shape(shape_ref));
  T value_list[] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 2, 3, 1, 0, 3, 2, 0, 1, 1, 0, 2, 1, 0, 1, 1, 2};

  tensor_fill_list(ref, value_list, array_size(value_list));

  EXPECT_LT(tensor_rel_error(ref, ret), 1e-7);
}


TEST_F(LayerConvTest, forward_from_picture) {
  conv_param_t conv_params = {1, 0};

  uint n = 1;
  uint img_sz = 3;
  uint c = 2;
  uint fltr_sz = 2;
  uint num_fil = 2;
  uint sz_out = 1 + (img_sz + 2 * conv_params.padding - fltr_sz) / conv_params.stride;

  uint const shape_x[] = {n, c, img_sz, img_sz}; // 2x3x4x4
  uint const shape_w[] = {num_fil, c, fltr_sz, fltr_sz}; // 3x3x4x4
  uint const shape_y[] = {n, num_fil, sz_out, sz_out}; // 2x3x2x2

  EXPECT_EQ(2, sz_out);

  T x_values[] = { 1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2 };
  tensor_t x = tensor_make(shape_x, dim_of_shape(shape_x));
  tensor_fill_list(x, x_values, array_size(x_values));
//  tensor_dump(x);

  tensor_t w = tensor_make(shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));
  lcache_t cache;
  make_empty_lcache(&cache);

  status_t ret = convolution_forward(x, w, &cache, conv_params, y);// forward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  tensor_t y_ref = tensor_make_alike(y);
  T value_list[] = { 6, 2, 3, 4, 3, 3, 7, 7 };
  tensor_fill_list(y_ref, value_list, array_size(value_list));

  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
  PINF("Consistent with expected results");
}


TEST_F(LayerConvTest, Construct) {
//  params.stride=2;
//  params.padding=1;
//
//  nr_img = 2;
//  sz_img = 4;
//  nr_in_channel = 3;
//  sz_filter = 4;
//  nr_filter =3;
//
//  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img}; // 2x3x4x4
//  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter, sz_filter}; // 3x3x4x4
//
//  x   = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
//  w   = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
//
//
//  make_empty_lcache(&cache);
}


TEST_F(LayerConvTest, Forward){
//
//  uint sz_out = 1 + (sz_img + 2 * params.padding - sz_filter) / params.stride;
//  EXPECT_EQ(2, sz_out);
//  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out}; // 2x3x2x2
//
//  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));
//
//  status_t ret = convolution_forward(x, w, &cache, params, y);// forward function should allocate and populate cache;
//  EXPECT_EQ(ret, S_OK);
//
//  tensor_t y_ref = tensor_make_alike(y);
//  T value_list[] = {0.02553947, 0.01900658, -0.03984868, -0.09432237,
//                    0.05964474,  0.09894079, 0.12641447,  0.19823684,
//                    0.09375, 0.178875,0.29267763,0.49079605,
//                    -0.36098684 -0.57783553,-0.67079605 -1.06632237,
//                    0.28701316, 0.42294079, 0.41630921,  0.6075,
//                    0.93501316,  1.42371711, 1.50341447,  2.28132237};
//
//  tensor_fill_list(y_ref, value_list, array_size(value_list));
//
//  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
//  PINF("Consistent with expected results");
}



TEST_F(LayerConvTest, DISABLED_Backward){
//  status_t ret;
//
//  uint sz_out = 1 + (sz_img + 2*params.padding - sz_filter)/params.stride;
//  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out}; // 4x2x5x5
//
//  // input for backward
//  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));
//
//  tensor_t dx = tensor_make_alike(x);
//  tensor_t dw = tensor_make_alike(w);
//
//  ret = convolution_backward(dx, dw, &cache, dy); // backward needs to call free_lcache(cache);
//  EXPECT_EQ(ret, S_OK);
//
//  /* II. Numerical check */
//  // I had to make this copy since lambda doesn't allow me to use global
//  // variable
//  tensor_t x_copy = tensor_make_copy(x);
//  tensor_t w_copy = tensor_make_copy(w);
//
//  tensor_t dx_ref = tensor_make_alike(x);
//  tensor_t dw_ref = tensor_make_alike(w);
//
//  // evaluate gradient of x
//  eval_numerical_gradient(
//      [w_copy](tensor_t const in, tensor_t out) {
//        convolution_forward(in, w_copy, NULL, params, out);
//      },
//      x, dy, dx_ref);
//  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
//  PINF("gradient check of x... is ok");
//
//  // evaluate gradient of w
//  eval_numerical_gradient(
//      [x_copy](tensor_t const in, tensor_t out) {
//        convolution_forward(x_copy, in , NULL, params, out);
//      },
//      w, dy, dw_ref);
//  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-7);
//  PINF("gradient check of w... is ok");
//
//  EXPECT_EQ(ret, S_OK);
}

// TODO: check with cudnn

TEST_F(LayerConvTest,CheckLcache){
//  EXPECT_EQ(cache.count, 0); // backward needs to call free_lcache(cache);
}



TEST_F(LayerConvTest, Destroy) {
//  tensor_destroy(x);
//  tensor_destroy(w);
//  free_lcache(&cache);
}


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
