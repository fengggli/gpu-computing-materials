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

  T w_values[] = {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2};
  tensor_t w = tensor_make(shape_w, dim_of_shape(shape_w));
  tensor_fill_list(w, w_values, array_size(w_values));

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

TEST_F(LayerConvTest, im2col_dot_operation) {
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

  uint const shape_ret[] = {ret.dim.dims[0], ret.dim.dims[1]};
  uint const shape_ref[] = {32};
  tensor_reshape_(&ret, shape_ref, dim_of_shape(shape_ref));

  tensor_t ref = tensor_make(shape_ref, dim_of_shape(shape_ref));
  T value_list[] = {1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 2, 3, 1, 0, 3, 2, 0, 1, 1, 0, 2, 1, 0, 1, 1, 2};
  tensor_fill_list(ref, value_list, array_size(value_list));

  EXPECT_LT(tensor_rel_error(ref, ret), 1e-7);

  tensor_reshape_(&ret, shape_ret, dim_of_shape(shape_ret));
}

TEST_F(LayerConvTest, test_tpose3012) {
  uint const shape_x[] = {2, 3, 4, 4}; // 2x3x4x4
  uint const shape_w[] = {3, 3, 4, 6}; // 3x3x4x6

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));

  tensor_t x_tpose_3012 = tensor_make_transpose_3012(x);
  tensor_t w_tpose_3012 = tensor_make_transpose_3012(w);

  tensor_t x_ref = tensor_make(shape_x, dim_of_shape(shape_x));
  T x3012[] = {-0.1, -0.07473684210526316, -0.04947368421052632, -0.024210526315789488, 0.0010526315789473606, 0.02631578947368421, 0.05157894736842103, 0.07684210526315788, 0.10210526315789473, 0.12736842105263158, 0.15263157894736842, 0.17789473684210524, 0.20315789473684207, 0.22842105263157894, 0.25368421052631573, 0.2789473684210526, 0.3042105263157895, 0.32947368421052625, 0.35473684210526313, 0.38, 0.4052631578947369, 0.43052631578947365, 0.4557894736842105, 0.4810526315789474, -0.0936842105263158, -0.06842105263157895, -0.04315789473684211, -0.01789473684210527, 0.007368421052631566, 0.0326315789473684, 0.05789473684210525, 0.0831578947368421, 0.10842105263157895, 0.13368421052631577, 0.15894736842105264, 0.18421052631578946, 0.20947368421052628, 0.23473684210526316, 0.26, 0.2852631578947368, 0.31052631578947365, 0.3357894736842105, 0.3610526315789474, 0.38631578947368417, 0.41157894736842104, 0.4368421052631579, 0.4621052631578948, 0.48736842105263156, -0.08736842105263158, -0.06210526315789475, -0.0368421052631579, -0.011578947368421064, 0.013684210526315785, 0.03894736842105262, 0.06421052631578947, 0.08947368421052632, 0.11473684210526314, 0.13999999999999999, 0.1652631578947368, 0.19052631578947368, 0.2157894736842105, 0.24105263157894738, 0.2663157894736842, 0.29157894736842105, 0.3168421052631579, 0.3421052631578947, 0.36736842105263157, 0.39263157894736844, 0.4178947368421053, 0.4431578947368421, 0.46842105263157896, 0.49368421052631584, -0.08105263157894738, -0.055789473684210535, -0.030526315789473693, -0.005263157894736845, 0.01999999999999999, 0.04526315789473684, 0.07052631578947369, 0.09578947368421051, 0.12105263157894736, 0.1463157894736842, 0.17157894736842103, 0.1968421052631579, 0.22210526315789472, 0.24736842105263154, 0.27263157894736845, 0.2978947368421052, 0.3231578947368421, 0.34842105263157896, 0.37368421052631573, 0.3989473684210526, 0.4242105263157895, 0.44947368421052636, 0.4747368421052631, 0.5};
  tensor_fill_list(x_ref, x3012, array_size(x3012));
  uint shape_tpose_x[] = {x.dim.dims[3], x.dim.dims[0], x.dim.dims[1], x.dim.dims[2]};
  tensor_reshape_(&x_ref, shape_tpose_x, dim_of_shape(shape_tpose_x));

  tensor_t w_ref = tensor_make(shape_w, dim_of_shape(shape_w));
  T w3012[] = {-0.2, -0.18604651162790697, -0.17209302325581396, -0.15813953488372096, -0.14418604651162792, -0.13023255813953488, -0.11627906976744187, -0.10232558139534885, -0.08837209302325583, -0.07441860465116279, -0.06046511627906978, -0.04651162790697677, -0.032558139534883734, -0.018604651162790697, -0.004651162790697688, 0.009302325581395321, 0.023255813953488358, 0.037209302325581395, 0.05116279069767443, 0.06511627906976741, 0.07906976744186045, 0.09302325581395349, 0.10697674418604647, 0.1209302325581395, 0.13488372093023254, 0.14883720930232558, 0.16279069767441862, 0.1767441860465116, 0.19069767441860463, 0.20465116279069767, 0.21860465116279065, 0.2325581395348837, 0.24651162790697673, 0.26046511627906976, 0.2744186046511628, 0.2883720930232558, -0.19767441860465118, -0.18372093023255814, -0.16976744186046513, -0.1558139534883721, -0.1418604651162791, -0.12790697674418605, -0.11395348837209304, -0.1, -0.086046511627907, -0.07209302325581396, -0.05813953488372095, -0.04418604651162791, -0.030232558139534904, -0.016279069767441867, -0.002325581395348858, 0.011627906976744179, 0.025581395348837188, 0.039534883720930225, 0.05348837209302326, 0.06744186046511624, 0.08139534883720928, 0.09534883720930232, 0.10930232558139535, 0.12325581395348834, 0.13720930232558137, 0.1511627906976744, 0.16511627906976745, 0.17906976744186043, 0.19302325581395346, 0.2069767441860465, 0.22093023255813954, 0.23488372093023252, 0.24883720930232556, 0.2627906976744186, 0.27674418604651163, 0.2906976744186046, -0.19534883720930235, -0.1813953488372093, -0.16744186046511628, -0.15348837209302327, -0.13953488372093026, -0.12558139534883722, -0.1116279069767442, -0.09767441860465118, -0.08372093023255815, -0.06976744186046513, -0.05581395348837209, -0.04186046511627908, -0.027906976744186074, -0.013953488372093037, 0.0, 0.01395348837209301, 0.02790697674418602, 0.041860465116279055, 0.05581395348837209, 0.06976744186046507, 0.08372093023255811, 0.09767441860465115, 0.11162790697674418, 0.12558139534883722, 0.1395348837209302, 0.15348837209302324, 0.16744186046511628, 0.18139534883720926, 0.1953488372093023, 0.20930232558139533, 0.22325581395348837, 0.2372093023255814, 0.2511627906976744, 0.2651162790697674, 0.27906976744186046, 0.29302325581395344, -0.1930232558139535, -0.17906976744186048, -0.16511627906976745, -0.15116279069767444, -0.1372093023255814, -0.12325581395348839, -0.10930232558139535, -0.09534883720930235, -0.08139534883720931, -0.0674418604651163, -0.05348837209302326, -0.03953488372093025, -0.025581395348837216, -0.011627906976744207, 0.0023255813953488302, 0.01627906976744184, 0.030232558139534876, 0.044186046511627886, 0.05813953488372092, 0.07209302325581396, 0.08604651162790694, 0.09999999999999998, 0.11395348837209301, 0.12790697674418605, 0.14186046511627903, 0.15581395348837207, 0.1697674418604651, 0.18372093023255814, 0.19767441860465113, 0.21162790697674416, 0.2255813953488372, 0.23953488372093024, 0.2534883720930232, 0.26744186046511625, 0.2813953488372093, 0.29534883720930233, -0.19069767441860466, -0.17674418604651165, -0.16279069767441862, -0.14883720930232558, -0.13488372093023257, -0.12093023255813955, -0.10697674418604652, -0.0930232558139535, -0.07906976744186048, -0.06511627906976747, -0.05116279069767443, -0.037209302325581395, -0.023255813953488386, -0.009302325581395376, 0.0046511627906976605, 0.018604651162790697, 0.03255813953488371, 0.046511627906976716, 0.06046511627906975, 0.07441860465116279, 0.08837209302325583, 0.10232558139534881, 0.11627906976744184, 0.13023255813953488, 0.14418604651162786, 0.1581395348837209, 0.17209302325581394, 0.18604651162790697, 0.2, 0.213953488372093, 0.22790697674418603, 0.24186046511627907, 0.25581395348837205, 0.2697674418604651, 0.2837209302325581, 0.29767441860465116, -0.18837209302325583, -0.1744186046511628, -0.16046511627906979, -0.14651162790697675, -0.13255813953488374, -0.1186046511627907, -0.1046511627906977, -0.09069767441860466, -0.07674418604651165, -0.06279069767441861, -0.0488372093023256, -0.034883720930232565, -0.020930232558139555, -0.0069767441860465185, 0.006976744186046491, 0.020930232558139528, 0.03488372093023254, 0.048837209302325574, 0.06279069767441858, 0.07674418604651162, 0.09069767441860466, 0.10465116279069764, 0.11860465116279068, 0.1325581395348837, 0.14651162790697675, 0.16046511627906973, 0.17441860465116277, 0.1883720930232558, 0.20232558139534884, 0.21627906976744182, 0.23023255813953486, 0.2441860465116279, 0.25813953488372093, 0.2720930232558139, 0.28604651162790695, 0.3};
  tensor_fill_list(w_ref, w3012, array_size(w3012));
  uint shape_tpose_w[] = {w.dim.dims[3], w.dim.dims[0], w.dim.dims[1], w.dim.dims[2]};
  tensor_reshape_(&w_ref, shape_tpose_w, dim_of_shape(shape_tpose_w));

  EXPECT_LT(tensor_rel_error(x_tpose_3012, x_ref), 1e-7);
  EXPECT_LT(tensor_rel_error(w_tpose_3012, w_ref), 1e-7);

  tensor_destroy(&x_tpose_3012);
  tensor_destroy(&x_ref);
  tensor_destroy(&w_tpose_3012);
  tensor_destroy(&w_ref);
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


  T w_values[] = {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2};
  tensor_t w = tensor_make(shape_w, dim_of_shape(shape_w));
  tensor_fill_list(w, w_values, array_size(w_values));

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

TEST_F(LayerConvTest, Forward){
  conv_param_t conv_params;

  conv_params.stride=2;
  conv_params.padding=1;

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 4;
  uint nr_filter = 3;

  uint sz_out = 1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;
  EXPECT_EQ(2, sz_out);

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img}; // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter, sz_filter}; // 3x3x4x4
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out}; // 2x3x2x2

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  lcache_t cache;
  make_empty_lcache(&cache);

  status_t ret = convolution_forward(x, w, &cache, conv_params, y);// forward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);
  tensor_dump(y);

  tensor_t y_ref = tensor_make_alike(y);
  T value_list[] =  {0.012401913875598115, -0.009877806404122181, -0.08387191755612806, -0.11092160471107837, 0.16027088700772907, 0.166610967979389, 0.17847626058152372, 0.1800463746779536, 0.30813986013986006, 0.3430997423629002, 0.4408244387191755, 0.47101435406698555, -0.8805358851674642, -0.9314354066985646, -1.0912889216047112, -1.1469584100110415, 0.6410835480309163, 0.618803827751196, 0.5448097165991902, 0.5177600294442398, 2.162702981229297, 2.1690430622009567, 2.1809083548030914, 2.1824784688995216};

  tensor_fill_list(y_ref, value_list, array_size(value_list));

  EXPECT_LT(tensor_rel_error(y_ref, y), 1e-7);
  PINF("Consistent with expected results");
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
//  EXPECT_EQ(cache.count, 0); // backward needs to pop all all caches and destroy them
}



TEST_F(LayerConvTest, Destroy) {
//  tensor_destroy(x);
//  tensor_destroy(w);
//  lcache_free_all(&cache);
}


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
