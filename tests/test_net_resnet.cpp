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
TEST_F(NetResnetTest, Destroy) { resnet_finalize(&model); }

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
