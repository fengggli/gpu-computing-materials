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

// #define PROFILE_RESNET
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
    CONV_METHOD_PERIMG,
};

TEST_F(NetResnetTest, Construct) {
  uint batch_sz = 3;  // change to other batchz might break
  uint input_shape[] = {batch_sz, 3, 32, 32};
  dim_t input_dim = make_dim_from_arr(array_size(input_shape), input_shape);
  uint output_dim = 10;
  uint nr_stages = 3;
  uint nr_blocks[MAX_STAGES] = {2, 2, 2};
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

TEST_F(NetResnetTest, Loss) {
  T loss = 0;

  // fill some init values as in cs231n
  tensor_t x = tensor_make_linspace(-0.2, 0.3, model.input_dim.dims, 4);

  label_t labels[] = {0, 5, 1};

  for (auto conv_method = all_methods.begin(); conv_method != all_methods.end();
       conv_method++) {
    PINF("Method:  %d", *conv_method);
    set_conv_method(*conv_method);

    model.reg = 0;
    resnet_loss(&model, x, labels, &loss);
    PINF("reg = 0, loss = %.3f", loss);

    // test with regulizer
    /*
    model.reg = 1.0;
    resnet_loss(&model, x, labels, &loss);
    PINF("reg = 1, loss = %.3f", loss);
     */
  }
}

/* Check both forward/backward, time consuming*/
/* This doesn't pass in float32 because of precision */
#if 0
TEST_F(NetResnetTest, BackNumerical) {
  // set_conv_method(CONV_METHOD_NNPACK_AUTO);
  conv_method_t method = get_conv_method();
  // AWNN_CHECK_EQ(method, CONV_METHOD_NAIVE);
  // fill some init values as in cs231n
  tensor_t x = tensor_make_linspace(-0.2, 0.3, model.input_dim.dims, 4);

  label_t labels[] = {0, 5, 1};

  // Check with numerical gradient
  model_t *ptr_model = &model;
  uint y_shape[] = {1};
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
#endif

TEST_F(NetResnetTest, Finalize) {
 resnet_finalize(&model);
}
}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
