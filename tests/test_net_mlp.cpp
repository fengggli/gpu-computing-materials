/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/net_mlp.h"
#include "awnn/weight_init.h"

#include "gtest/gtest.h"
#include "test_util.h"

namespace {

// The fixture for testing class Foo.
class NetMLPTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  NetMLPTest() {
    // You can do set-up work for each test here.
  }

  ~NetMLPTest() override {
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
  static model_t model;
};

// input of forward
model_t NetMLPTest::model;

TEST_F(NetMLPTest, Construct) {
  uint batch_sz = 3;
  uint input_dim = 5;
  uint output_dim = 7;
  uint nr_hidden_layers = 1;
  uint hidden_dims[] = {50};
  T reg = 0;

  mlp_init(&model, batch_sz, input_dim, output_dim, nr_hidden_layers,
           hidden_dims, reg);

  EXPECT_EQ((void *)0, (void *)net_get_param(model.list_all_params,
                                             "W3"));  // unexisting param
  EXPECT_NE((void *)0,
            (void *)net_get_param(model.list_all_params, "fc0.weight"));
}

/* Interference-only forward*/
TEST_F(NetMLPTest, ForwardInferOnly) {
  uint x_shape[] = {model.max_batch_sz, model.input_dim};
  tensor_t w0 = net_get_param(model.list_all_params, "fc0.weight")->data;
  tensor_t b0 = net_get_param(model.list_all_params, "fc0.bias")->data;
  tensor_t w1 = net_get_param(model.list_all_params, "fc1.weight")->data;
  tensor_t b1 = net_get_param(model.list_all_params, "fc1.bias")->data;

  // fill some init values as in cs231n
  weight_init_linspace(w0, -0.7, 0.3);
  weight_init_linspace(b0, -0.1, 0.9);
  weight_init_linspace(w1, -0.3, 0.4);
  weight_init_linspace(b1, -0.9, 0.1);

  tensor_t x = tensor_make_linspace(-5.5, 4.5, x_shape, 2);
  tensor_t score = mlp_scores(&model, x);

  tensor_t score_ref = tensor_make_alike(score);

  // values is different from  assignment in cs231n
  // In the assignment there is a tranpose of X
  T value_list[] = {3.11899925, 3.87793183, 4.6368644,  5.39579698, 6.15472956,
                    6.91366214, 7.67259472, 5.74919872, 6.14996511, 6.5507315,
                    6.95149789, 7.35226428, 7.75303067, 8.15379705, 0.50899227,
                    0.6837731,  0.85855392, 1.03333475, 1.20811557, 1.3828964,
                    1.55767723};
  /*  T value_list[] = {11.53165108, 12.2917344,  13.05181771, 13.81190102,*/
  // 14.57198434, 15.33206765, 16.09215096, 12.05769098,
  // 12.74614105, 13.43459113, 14.1230412,  14.81149128,
  // 15.49994135, 16.18839143, 12.58373087, 13.20054771,
  // 13.81736455, 14.43418138, 15.05099822, 15.66781506,
  /*16.2846319};*/
  tensor_fill_list(score_ref, value_list, dim_of_shape(value_list));

  EXPECT_LT(tensor_rel_error(score_ref, score), 1e-7);
  if (tensor_rel_error(score_ref, score) > 1e-7) {
    tensor_dump(score);
    tensor_dump(score_ref);
  }
}

/* Check both forward/backward*/
TEST_F(NetMLPTest, DISABLED_Loss) {
  T loss = 0;
  uint x_shape[] = {model.max_batch_sz, model.input_dim};
  tensor_t x = tensor_make_linspace(-0.1, 0.5, x_shape, 2);
  label_t *labels = label_make_random(model.max_batch_sz, model.output_dim);

  mlp_loss(&model, x, labels, &loss);
  label_destroy(labels);
}

TEST_F(NetMLPTest, Destroy) { mlp_finalize(&model); }

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
