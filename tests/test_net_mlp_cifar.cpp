/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/data_utils.h"
#include "awnn/net_mlp.h"
#include "awnn/solver.h"
#include "awnn/weight_init.h"
#include "utils/data_cifar.h"

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
  uint batch_sz = 25;
  uint input_dim = 3 * 32 * 32;
  uint output_dim = 10;
  uint nr_hidden_layers = 2;
  uint hidden_dims[] = {100, 100};
  T reg = 0;

  mlp_init(&model, batch_sz, input_dim, output_dim, nr_hidden_layers,
           hidden_dims, reg);

  EXPECT_EQ((void *)0, (void *)net_get_param(model.list_all_params,
                                             "W3"));  // unexisting param
  EXPECT_NE((void *)0,
            (void *)net_get_param(model.list_all_params, "fc0.weight"));
}

TEST_F(NetMLPTest, CifarTest) {
  data_loader_t loader;
  status_t ret = cifar_open(&loader, CIFAR_PATH);
  EXPECT_EQ(S_OK, ret);

  // overfit small data;
  uint train_sz = 50;
  uint val_sz = 1000;
  T learning_rate = 0.1;

  EXPECT_EQ(S_OK, cifar_split_train(&loader, train_sz, val_sz));

  uint batch_sz = 25;  // this can be 50 000/50 = 1000 batches
  uint nr_epoches = 1;

  uint iterations_per_epoch = train_sz / batch_sz;
  if (iterations_per_epoch == 0) iterations_per_epoch = 1;
  uint nr_iterations = nr_epoches * iterations_per_epoch;

  tensor_t x;
  label_t *labels;
  T loss = 0;

  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    uint cur_epoch = iteration / iterations_per_epoch;
    uint cur_batch = iteration % iterations_per_epoch;

    PINF("[Epoch %d, Iteration %u/%u]", cur_epoch, cur_batch,
         iterations_per_epoch);
    EXPECT_EQ(batch_sz,
              get_train_batch(&loader, &x, &labels, cur_batch, batch_sz));

    mlp_loss(&model, x, labels, &loss);

    PINF("Loss %.2f", loss);

    param_t *p_param;
    // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
    list_for_each_entry(p_param, model.list_all_params, list) {
      PINF("updating %s...", p_param->name);
      // sgd
      sgd_update(p_param, learning_rate);
    }
  }
  cifar_close(&loader);
}

TEST_F(NetMLPTest, Destroy) { mlp_finalize(&model); }

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
