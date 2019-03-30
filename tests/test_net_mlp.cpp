/*
/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/net_mlp.h"

#include "test_util.h"
#include "gtest/gtest.h"

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
  uint batch_sz=64;
  uint input_dim = 3*32*32;
  uint output_dim = 10;
  uint nr_hidden_layers = 1;
  uint hidden_dims[] = {50};
  T reg = 0;

  mlp_init(&model, batch_sz, input_dim, output_dim, nr_hidden_layers, hidden_dims, reg);

  EXPECT_EQ((void *)0, (void *)net_get_param(model.list_all_params, "W3")); // unexisting param
  EXPECT_NE((void *)0, (void *)net_get_param(model.list_all_params, "W1"));
}

TEST_F(NetMLPTest, Destroy) {
  mlp_finalize(&model);
}


}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

