/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/data_utils.h"
#include "awnn/net_resnet.h"
#include "awnn/solver.h"
#include "utils/debug.h"
#include "utils/weight_init.h"

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

TEST_F(NetResnetTest, Construct) {
  uint batch_sz = 3;
  uint input_shape[] = {batch_sz, 3, 32, 32};
  dim_t input_dim = make_dim_from_arr(array_size(input_shape), input_shape);
  uint output_dim = 10;
  uint nr_stages = 1;
  uint nr_blocks[MAX_STAGES] = {2};
  T reg = 0;
  normalize_method_t normalize_method = NORMALIZE_NONE;  // no batchnorm now

  resnet_init(&model, input_dim, output_dim, nr_stages, nr_blocks, reg,
              normalize_method);

  EXPECT_EQ((void *)0, (void *)net_get_param(model.list_all_params,
                                             "W3"));  // unexisting param
  /*
  EXPECT_NE((void *)0,
            (void *)net_get_param(model.list_all_params, "fc0.weight"));
            */
}
TEST_F(NetResnetTest, ForwardInferOnly) {
  tensor_t x = tensor_make_linspace(-5.5, 4.5, model.input_dim.dims, 4);

  tensor_t score = resnet_forward_infer(&model, x);
}

/* Check both forward/backward*/
TEST_F(NetResnetTest, Loss) {
  T loss = 0;

  tensor_t x = tensor_make_linspace(-5.5, 4.5, model.input_dim.dims, 4);
  label_t labels[] = {0, 5, 1};

  uint nr_iterations = 10;
  for(uint i = 0; i< nr_iterations; i++) {
    resnet_loss(&model, x, labels, &loss);
    // PINF("Loss without regulizer: %.3f", loss);
  }


  /*
  // test with regulizer
  model.reg = 1.0;
  resnet_loss(&model, x, labels, &loss);
  PINF("Loss with regulizer: %.3f", loss);
  // EXPECT_NEAR(loss, 26.11873099, 1e-7);
  PINF("Forward/backward finished");

  // Check with numerical gradient
  model_t model_copy = model;
  uint y_shape[] = {1};
  tensor_t dy = tensor_make_ones(y_shape, dim_of_shape(y_shape));
  dy.data[0] = 1.0;  // the y is the loss, no upper layer

  param_t *p_param;
  // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
  list_for_each_entry(p_param, model.list_all_params, list) {
    tensor_t param = p_param->data;
    tensor_t dparam = p_param->diff;
    tensor_t dparam_ref = tensor_make_alike(param);
    eval_numerical_gradient(
        [model_copy, x, labels](tensor_t const, tensor_t out) {
          T *ptr_loss = &out.data[0];
          resnet_loss(&model_copy, x, labels, ptr_loss);
        },
        param, dy, dparam_ref);

    EXPECT_LT(tensor_rel_error(dparam_ref, dparam), 1e-7);
    tensor_destroy(&dparam_ref);
    PINF("Gradient check of %s passed", p_param->name);
  }
  */
}

TEST_F(NetResnetTest, Destroy) { resnet_finalize(&model); }

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
