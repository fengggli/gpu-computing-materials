/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/data_utils.h"
#include "awnn/layer_conv.h"
#include "awnn/net_resnet.h"
#include "awnn/solver.h"
#include "utils/debug.h"
#include "utils/weight_init.h"

#include <vector>
#include "gtest/gtest.h"
#include "test_util.h"

namespace {

// The fixture for testing class Foo.
class NetResnetTest : public ::testing::Test {
 protected:
  // Objects declared here can be used by all tests.
  static model_t model;
};

// input of forward
model_t NetResnetTest::model;
std::vector<conv_method_t> all_methods = {
    CONV_METHOD_NNPACK_AUTO, CONV_METHOD_NNPACK_ft8x8,
    CONV_METHOD_NNPACK_ft16x16,
    // CONV_METHOD_NNPACK_wt8x8,  // returns 26
    // CONV_METHOD_NNPACK_implicit_gemm,  // returns 26
    // CONV_METHOD_NNPACK_direct, // returns 26
    CONV_METHOD_NNPACK_REF, CONV_METHOD_NAIVE};

TEST_F(NetResnetTest, Construct) {
  int batch_sz = 3;
  int input_shape[] = {batch_sz, 3, 32, 32};
  dim_t input_dim = make_dim_from_arr(array_size(input_shape), input_shape);
  int output_dim = 10;
  int nr_stages = 1;
  int nr_blocks[MAX_STAGES] = {1};
  T reg = 0;
  normalize_method_t normalize_method = NORMALIZE_NONE;  // no batchnorm now

  resnet_init(&model, input_dim, output_dim, nr_stages, nr_blocks, reg,
              normalize_method);

  EXPECT_EQ((void *)0, (void *)net_get_param(model.list_all_params,
                                             "W3"));  // unexisting param
  /*
  EXPECT_NE((void *)0, (void *)net_get_param(model.list_all_params,
  "fc0.weight"));
            */
}
TEST_F(NetResnetTest, ForwardInferOnly) {
  tensor_t w0 = net_get_param(model.list_all_params, "conv1.weight")->data;
  tensor_t w1 =
      net_get_param(model.list_all_params, "layer1.1.conv1.weight")->data;
  tensor_t w2 =
      net_get_param(model.list_all_params, "layer1.1.conv2.weight")->data;
  tensor_t w_fc = net_get_param(model.list_all_params, "fc.weight")->data;
  tensor_t b_fc = net_get_param(model.list_all_params, "fc.bias")->data;

  // fill some init values as in cs231n
  tensor_t x = tensor_make_linspace(-0.2, 0.3, model.input_dim.dims, 4);

  weight_init_linspace(w0, -0.7, 0.3);
  weight_init_linspace(w1, -0.7, 0.3);
  weight_init_linspace(w2, -0.7, 0.3);
  weight_init_linspace(w_fc, -0.7, 0.3);
  weight_init_linspace(b_fc, -0.7, 0.3);

  tensor_t score = resnet_forward_infer(&model, x);
  tensor_t score_ref = tensor_make_alike(score);

  double value_list[] = {
      81.32275804, 84.57773448, 87.83271092,  91.08768737,  94.34266381,
      97.59764025, 100.8526167, 104.10759314, 107.36256958, 110.61754603,
      8.74922391,  9.22185456,  9.69448521,   10.16711586,  10.63974651,
      11.11237715, 11.5850078,  12.05763845,  12.5302691,   13.00289975,
      37.25164058, 38.81409578, 40.37655098,  41.93900617,  43.50146137,
      45.06391657, 46.62637176, 48.18882696,  49.75128216,  51.31373735};

  tensor_fill_list(score_ref, value_list, dim_of_shape(value_list));

  EXPECT_LT(tensor_rel_error(score_ref, score), 1e-7);
  if (tensor_rel_error(score_ref, score) > 1e-7) {
    tensor_dump(score);
    tensor_dump(score_ref);
  }
  tensor_destroy(&x);
}
TEST_F(NetResnetTest, Loss) {
  T loss = 0;
  std::vector<conv_method_t> all_methods = {
      CONV_METHOD_NNPACK_AUTO, CONV_METHOD_NNPACK_ft8x8,
      CONV_METHOD_NNPACK_ft16x16,
      // CONV_METHOD_NNPACK_wt8x8,  // returns 26
      // CONV_METHOD_NNPACK_implicit_gemm,  // returns 26
      // CONV_METHOD_NNPACK_direct, // returns 26
      CONV_METHOD_NNPACK_REF, CONV_METHOD_NAIVE};

  // fill some init values as in cs231n
  tensor_t x = tensor_make_linspace(-0.2, 0.3, model.input_dim.dims, 4);

  label_t labels[] = {0, 5, 1};

  for (auto conv_method = all_methods.begin(); conv_method != all_methods.end();
       conv_method++) {
    PINF("Method:  %d", *conv_method);
    set_conv_method(*conv_method);

    model.reg = 0;
    resnet_loss(&model, x, labels, &loss);
    EXPECT_NEAR(loss, 14.975702563, 1e-3);

    // test with regulizer
    model.reg = 1.0;
    resnet_loss(&model, x, labels, &loss);
    EXPECT_NEAR(loss, 335.9764923, 1e-3);
    PINF("Loss checked");
  }
}

/* Check both forward/backward, time consuming*/
TEST_F(NetResnetTest, DISABLED_BackNumerical) {
  // fill some init values as in cs231n
  tensor_t x = tensor_make_linspace(-0.2, 0.3, model.input_dim.dims, 4);

  label_t labels[] = {0, 5, 1};

  // Check with numerical gradient
  model_t *ptr_model = &model;
  int y_shape[] = {1};
  tensor_t dy = tensor_make_ones(y_shape, dim_of_shape(y_shape));
  dy.data[0] = 1.0;  // the y is the loss, no upper layer

  model.reg = 1.0;
  param_t *p_param;
  // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
  list_for_each_entry(p_param, model.list_all_params, list) {
    tensor_t param = p_param->data;
    tensor_t dparam = p_param->diff;
    tensor_t dparam_ref = tensor_make_alike(param);
    eval_numerical_gradient(
        [ptr_model, x, labels](tensor_t const, tensor_t out) {
          T *ptr_loss = &out.data[0];
          resnet_loss(ptr_model, x, labels, ptr_loss);
        },
        param, dy, dparam_ref);

    EXPECT_LT(tensor_rel_error(dparam_ref, dparam), 1e-5);
    tensor_destroy(&dparam_ref);
    PINF("Gradient check of %s passed", p_param->name);
  }
}

/* Check both forward/backward*/
TEST_F(NetResnetTest, Measure_auto) {
  for (auto conv_method = all_methods.begin(); conv_method != all_methods.end();
       conv_method++) {
    PINF("Method:  %d", *conv_method);
    set_conv_method(*conv_method);

    for (int i = 0; i < 3; i++) {
      T loss = 0;

      tensor_t x = tensor_make_linspace(-5.5, 4.5, model.input_dim.dims, 4);
      label_t labels[] = {0, 5, 1};

      time_point_t t_begin, t_end;
      get_cur_time(t_begin);
      int nr_iterations = 100;
      for (int i = 0; i < nr_iterations; i++) {
        resnet_loss(&model, x, labels, &loss);
        // PINF("Loss without regulizer: %.3f", loss);
      }
      get_cur_time(t_end);
      print_time_in_ms(t_begin, t_end);
    }
  }
}

TEST_F(NetResnetTest, Destroy) { resnet_finalize(&model); }

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
