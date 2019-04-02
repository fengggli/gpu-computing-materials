#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

#include "awnn/common.h"

namespace {

// The fixture for testing class Foo.
class TensorOpTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  TensorOpTest() { // You can do set-up work for each test here.
  }

  ~TensorOpTest() override {
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
  // static tensor_t t;
};


TEST_F(TensorOpTest, test_tpose1230) {
  uint const shape_x[] = {2, 3, 4, 4}; // 2x3x4x4
  uint const shape_w[] = {3, 3, 4, 6}; // 3x3x4x6

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));

  tensor_t x_tpose_1230 = tensor_make_transpose_1230(x);
  tensor_t w_tpose_1230 = tensor_make_transpose_1230(w);

  tensor_t x_ref = tensor_make(shape_x, dim_of_shape(shape_x));
  T x1230[] = {-0.1, 0.20315789473684207, -0.0936842105263158, 0.20947368421052628, -0.08736842105263158, 0.2157894736842105, -0.08105263157894738, 0.22210526315789472, -0.07473684210526316, 0.22842105263157894, -0.06842105263157895, 0.23473684210526316, -0.06210526315789475, 0.24105263157894738, -0.055789473684210535, 0.24736842105263154, -0.04947368421052632, 0.25368421052631573, -0.04315789473684211, 0.26, -0.0368421052631579, 0.2663157894736842, -0.030526315789473693, 0.27263157894736845, -0.024210526315789488, 0.2789473684210526, -0.01789473684210527, 0.2852631578947368, -0.011578947368421064, 0.29157894736842105, -0.005263157894736845, 0.2978947368421052, 0.0010526315789473606, 0.3042105263157895, 0.007368421052631566, 0.31052631578947365, 0.013684210526315785, 0.3168421052631579, 0.01999999999999999, 0.3231578947368421, 0.02631578947368421, 0.32947368421052625, 0.0326315789473684, 0.3357894736842105, 0.03894736842105262, 0.3421052631578947, 0.04526315789473684, 0.34842105263157896, 0.05157894736842103, 0.35473684210526313, 0.05789473684210525, 0.3610526315789474, 0.06421052631578947, 0.36736842105263157, 0.07052631578947369, 0.37368421052631573, 0.07684210526315788, 0.38, 0.0831578947368421, 0.38631578947368417, 0.08947368421052632, 0.39263157894736844, 0.09578947368421051, 0.3989473684210526, 0.10210526315789473, 0.4052631578947369, 0.10842105263157895, 0.41157894736842104, 0.11473684210526314, 0.4178947368421053, 0.12105263157894736, 0.4242105263157895, 0.12736842105263158, 0.43052631578947365, 0.13368421052631577, 0.4368421052631579, 0.13999999999999999, 0.4431578947368421, 0.1463157894736842, 0.44947368421052636, 0.15263157894736842, 0.4557894736842105, 0.15894736842105264, 0.4621052631578948, 0.1652631578947368, 0.46842105263157896, 0.17157894736842103, 0.4747368421052631, 0.17789473684210524, 0.4810526315789474, 0.18421052631578946, 0.48736842105263156, 0.19052631578947368, 0.49368421052631584, 0.1968421052631579, 0.5};
  tensor_fill_list(x_ref, x1230, array_size(x1230));
  uint shape_tpose_x[] = {x.dim.dims[1], x.dim.dims[2], x.dim.dims[3], x.dim.dims[0]};
  tensor_reshape_(&x_ref, shape_tpose_x, dim_of_shape(shape_tpose_x));

  tensor_t w_ref = tensor_make(shape_w, dim_of_shape(shape_w));
  T w1230[] = {-0.2, -0.032558139534883734, 0.13488372093023254, -0.19767441860465118, -0.030232558139534904, 0.13720930232558137, -0.19534883720930235, -0.027906976744186074, 0.1395348837209302, -0.1930232558139535, -0.025581395348837216, 0.14186046511627903, -0.19069767441860466, -0.023255813953488386, 0.14418604651162786, -0.18837209302325583, -0.020930232558139555, 0.14651162790697675, -0.18604651162790697, -0.018604651162790697, 0.14883720930232558, -0.18372093023255814, -0.016279069767441867, 0.1511627906976744, -0.1813953488372093, -0.013953488372093037, 0.15348837209302324, -0.17906976744186048, -0.011627906976744207, 0.15581395348837207, -0.17674418604651165, -0.009302325581395376, 0.1581395348837209, -0.1744186046511628, -0.0069767441860465185, 0.16046511627906973, -0.17209302325581396, -0.004651162790697688, 0.16279069767441862, -0.16976744186046513, -0.002325581395348858, 0.16511627906976745, -0.16744186046511628, 0.0, 0.16744186046511628, -0.16511627906976745, 0.0023255813953488302, 0.1697674418604651, -0.16279069767441862, 0.0046511627906976605, 0.17209302325581394, -0.16046511627906979, 0.006976744186046491, 0.17441860465116277, -0.15813953488372096, 0.009302325581395321, 0.1767441860465116, -0.1558139534883721, 0.011627906976744179, 0.17906976744186043, -0.15348837209302327, 0.01395348837209301, 0.18139534883720926, -0.15116279069767444, 0.01627906976744184, 0.18372093023255814, -0.14883720930232558, 0.018604651162790697, 0.18604651162790697, -0.14651162790697675, 0.020930232558139528, 0.1883720930232558, -0.14418604651162792, 0.023255813953488358, 0.19069767441860463, -0.1418604651162791, 0.025581395348837188, 0.19302325581395346, -0.13953488372093026, 0.02790697674418602, 0.1953488372093023, -0.1372093023255814, 0.030232558139534876, 0.19767441860465113, -0.13488372093023257, 0.03255813953488371, 0.2, -0.13255813953488374, 0.03488372093023254, 0.20232558139534884, -0.13023255813953488, 0.037209302325581395, 0.20465116279069767, -0.12790697674418605, 0.039534883720930225, 0.2069767441860465, -0.12558139534883722, 0.041860465116279055, 0.20930232558139533, -0.12325581395348839, 0.044186046511627886, 0.21162790697674416, -0.12093023255813955, 0.046511627906976716, 0.213953488372093, -0.1186046511627907, 0.048837209302325574, 0.21627906976744182, -0.11627906976744187, 0.05116279069767443, 0.21860465116279065, -0.11395348837209304, 0.05348837209302326, 0.22093023255813954, -0.1116279069767442, 0.05581395348837209, 0.22325581395348837, -0.10930232558139535, 0.05813953488372092, 0.2255813953488372, -0.10697674418604652, 0.06046511627906975, 0.22790697674418603, -0.1046511627906977, 0.06279069767441858, 0.23023255813953486, -0.10232558139534885, 0.06511627906976741, 0.2325581395348837, -0.1, 0.06744186046511624, 0.23488372093023252, -0.09767441860465118, 0.06976744186046507, 0.2372093023255814, -0.09534883720930235, 0.07209302325581396, 0.23953488372093024, -0.0930232558139535, 0.07441860465116279, 0.24186046511627907, -0.09069767441860466, 0.07674418604651162, 0.2441860465116279, -0.08837209302325583, 0.07906976744186045, 0.24651162790697673, -0.086046511627907, 0.08139534883720928, 0.24883720930232556, -0.08372093023255815, 0.08372093023255811, 0.2511627906976744, -0.08139534883720931, 0.08604651162790694, 0.2534883720930232, -0.07906976744186048, 0.08837209302325583, 0.25581395348837205, -0.07674418604651165, 0.09069767441860466, 0.25813953488372093, -0.07441860465116279, 0.09302325581395349, 0.26046511627906976, -0.07209302325581396, 0.09534883720930232, 0.2627906976744186, -0.06976744186046513, 0.09767441860465115, 0.2651162790697674, -0.0674418604651163, 0.09999999999999998, 0.26744186046511625, -0.06511627906976747, 0.10232558139534881, 0.2697674418604651, -0.06279069767441861, 0.10465116279069764, 0.2720930232558139, -0.06046511627906978, 0.10697674418604647, 0.2744186046511628, -0.05813953488372095, 0.10930232558139535, 0.27674418604651163, -0.05581395348837209, 0.11162790697674418, 0.27906976744186046, -0.05348837209302326, 0.11395348837209301, 0.2813953488372093, -0.05116279069767443, 0.11627906976744184, 0.2837209302325581, -0.0488372093023256, 0.11860465116279068, 0.28604651162790695, -0.04651162790697677, 0.1209302325581395, 0.2883720930232558, -0.04418604651162791, 0.12325581395348834, 0.2906976744186046, -0.04186046511627908, 0.12558139534883722, 0.29302325581395344, -0.03953488372093025, 0.12790697674418605, 0.29534883720930233, -0.037209302325581395, 0.13023255813953488, 0.29767441860465116, -0.034883720930232565, 0.1325581395348837, 0.3};
  tensor_fill_list(w_ref, w1230, array_size(w1230));
  uint shape_tpose_w[] = {w.dim.dims[1], w.dim.dims[2], w.dim.dims[3], w.dim.dims[0]};
  tensor_reshape_(&w_ref, shape_tpose_w, dim_of_shape(shape_tpose_w));

  EXPECT_LT(tensor_rel_error(x_tpose_1230, x_ref), 1e-7);
  EXPECT_LT(tensor_rel_error(w_tpose_1230, w_ref), 1e-7);

  tensor_destroy(&x_tpose_1230);
  tensor_destroy(&x_ref);
  tensor_destroy(&w_tpose_1230);
  tensor_destroy(&w_ref);
}

TEST_F(TensorOpTest, test_tpose3012) {
  uint const shape_p[] = {1, 2, 3, 3}; // 1x2x3x3
  uint const shape_x[] = {2, 3, 4, 4}; // 2x3x4x4
  uint const shape_w[] = {3, 3, 4, 6}; // 3x3x4x6

  tensor_t p = tensor_make(shape_p, dim_of_shape(shape_p));
  T p_vals[] = {1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2};
  tensor_fill_list(p, p_vals, array_size(p_vals));
  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));

  tensor_t p_tpose_3012 = tensor_make_transpose_3012(p);
  tensor_t x_tpose_3012 = tensor_make_transpose_3012(x);
  tensor_t w_tpose_3012 = tensor_make_transpose_3012(w);

  tensor_t p_ref = tensor_make(shape_p, dim_of_shape(shape_p));
  T p3012[] = {1, 0, 1, 2, 1, 2, 0, 1, 1, 3, 0, 1, 1, 0, 1, 2, 1, 2};
  tensor_fill_list(p_ref, p3012, array_size(p3012));
  uint shape_tpose_p[] = {p.dim.dims[3], p.dim.dims[0], p.dim.dims[1], p.dim.dims[2]};
  tensor_reshape_(&p_ref, shape_tpose_p, dim_of_shape(shape_tpose_p));

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

  EXPECT_LT(tensor_rel_error(p_tpose_3012, p_ref), 1e-7);
  EXPECT_LT(tensor_rel_error(x_tpose_3012, x_ref), 1e-7);
  EXPECT_LT(tensor_rel_error(w_tpose_3012, w_ref), 1e-7);

  tensor_destroy(&x_tpose_3012);
  tensor_destroy(&x_ref);
  tensor_destroy(&w_tpose_3012);
  tensor_destroy(&w_ref);
}

// Tests that the Foo::Bar() method does Abc.
TEST_F(TensorOpTest, DotWrongInput) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 4};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));

  uint const shape3[] = {3, 4};
  out = tensor_make_patterned(shape3, dim_of_shape(shape3));

  EXPECT_TRUE(S_ERR == tensor_matmul(in1, in2, out));
}

TEST_F(TensorOpTest, Dot) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));
  // tensor_dump(in1);

  uint const shape2[] = {3, 2};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));
  // tensor_dump(in2);

  uint const shape3[] = {2, 2};
  out = tensor_make_patterned(shape3, dim_of_shape(shape3));

  // int correct_result[] = {10, 13, 28, 40};

  EXPECT_EQ(S_OK, tensor_matmul(in1, in2, out));
  // tensor_dump(out);
  EXPECT_EQ(out.data[0], 10);
  EXPECT_EQ(out.data[1], 13);
  EXPECT_EQ(out.data[2], 28);
  EXPECT_EQ(out.data[3], 40);
}

TEST_F(TensorOpTest, PLUS) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));
  // tensor_dump(in1);

  uint const shape2[] = {2, 3};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));
  // tensor_dump(in2);

  uint const shape3[] = {2, 3};
  out = tensor_make(shape3, dim_of_shape(shape3));

  EXPECT_EQ(S_OK, tensor_add_sameshape(in1, in2, out));
  EXPECT_EQ(out.data[0], 0);
  EXPECT_EQ(out.data[1], 2);
  EXPECT_EQ(out.data[2], 4);
  EXPECT_EQ(out.data[3], 6);
  EXPECT_EQ(out.data[4], 8);
  EXPECT_EQ(out.data[5], 10);
}

TEST_F(TensorOpTest, PLUS_INPLACE) {
  tensor_t from, to;
  uint const shape1[] = {2, 3};
  from = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 3};
  to = tensor_make_patterned(shape2, dim_of_shape(shape2));

  EXPECT_EQ(S_OK, tensor_elemwise_op_inplace(to, from, TENSOR_OP_ADD));

  EXPECT_EQ(to.data[0], 0);
  EXPECT_EQ(to.data[1], 2);
  EXPECT_EQ(to.data[2], 4);
  EXPECT_EQ(to.data[3], 6);
  EXPECT_EQ(to.data[4], 8);
  EXPECT_EQ(to.data[5], 10);
}

TEST_F(TensorOpTest, RESHAPE) {
  tensor_t t;
  uint const shape1[] = {2, 3, 4};
  t = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 13}; // this shall gives a error
  uint const shape3[] = {2, 3, 2, 2};
  uint const shape4[] = {2, 12};
  uint const shape5[] = {24};

  EXPECT_EQ(S_BAD_DIM, tensor_reshape_(&t, shape2, 2));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape3, 4));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape4, 2));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape5, 1));

  EXPECT_EQ(t.dim.dims[0], 24);
  EXPECT_EQ(t.dim.dims[1], 0);
}

TEST_F(TensorOpTest, AddVector) {
  tensor_t t, v;
  uint const shape1[] = {2, 3, 4, 5};
  uint const shape2[] = {5};
  t = tensor_make_patterned(shape1, dim_of_shape(shape1));
  v = tensor_make_patterned(shape2, dim_of_shape(shape2));

  EXPECT_EQ(S_OK, tensor_add_vector_inplace(t, v));

  EXPECT_EQ(t.data[0], 0);
  EXPECT_EQ(t.data[1], 2);
  EXPECT_EQ(t.data[5], 5);

  tensor_destroy(&t);
  tensor_destroy(&v);
}

TEST_F(TensorOpTest, tensor_reshape_flat_) {
  tensor_t t;
  uint const shape1[] = {2, 3, 4, 5};
  t = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint original_capacity_of_t = tensor_get_capacity(t);

  tensor_reshape_flat_(&t);

  uint new_capacity_of_t = tensor_get_capacity(t);

  ASSERT_EQ(original_capacity_of_t, new_capacity_of_t);

  int i;
  for (i = 0; i < MAX_DIM - 1; ++i) {
    ASSERT_EQ(t.dim.dims[i], 1);
  }
  ASSERT_EQ(t.dim.dims[i], original_capacity_of_t);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test0) {
  uint const shape[] = { 1, 1, 1, 1 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 0;
  float pad_val = 0;

  // 1 x 1 x 1 x 1 -> 1 x 1 x 0 x 0
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

  tensor_dump(in);
  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test1) {
  uint const shape[] = { 1, 1, 1, 1 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 1 x 1 -> 1 x 1 x 3 x 3
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test2) {
  uint const shape[] = { 1, 1, 1, 1 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 1 x 1 -> 1 x 1 x 3 x 3
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);
//
//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test3) {
  uint const shape[] = { 1, 1, 1, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 1 x 1 -> 1 x 1 x 3 x 3
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test4) {
  uint const shape[] = { 1, 1, 2, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 2 x 2 -> 1 x 1 x 4 x 4
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test5) {
  uint const shape[] = { 1, 1, 2, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 2 x 2 -> 1 x 1 x 6 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test6) {
  uint const shape[] = { 1, 1, 3, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 5 x 4
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test7) {
  uint const shape[] = { 1, 1, 3, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test8) {
  uint const shape[] = { 1, 1, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test9) {
  uint const shape[] = { 1, 2, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test10) {
  uint const shape[] = { 2, 1, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test11) {
  uint const shape[] = { 2, 2, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(&in);
  tensor_destroy(&padded_in);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
