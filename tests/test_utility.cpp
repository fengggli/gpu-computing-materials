/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/logging.h"
#include "awnn/tensor.h"
#include "test_util.h"
#include "gtest/gtest.h"

namespace {

// The fixture for testing class Foo.
class UtilityTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  UtilityTest() {
    // You can do set-up work for each test here.
  }

  ~UtilityTest() override {
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
};

TEST_F(UtilityTest, NumericalGradientSimple) {
  int const shape[] = {2, 3}; // a 2x3 matrix
  tensor_t x = tensor_make_linspace(1, 6, shape, dim_of_shape(shape));
  tensor_t dx = tensor_make(shape, dim_of_shape(shape));

  int const y_shape[] = {1};
  T val_dy = 3.0;
  tensor_t dy = tensor_make_scalar(y_shape, dim_of_shape(y_shape), val_dy);

  // the gradient in each xi will by val_dy
  auto func_add = [](tensor_t const input, tensor_t output) {
    output.data[0] = 0;
    for (int i = 0; i < tensor_get_capacity(input); i++) {
      PLOG("[+ %.7f at %u]", input.data[i], i);
      output.data[0] += input.data[i];
    }
    PLOG("----------");
  };

  eval_numerical_gradient(func_add, x, dy, dx, 1e-5);

  for (int i = 0; i < tensor_get_capacity(x); i++)
    EXPECT_NEAR(val_dy, dx.data[i], 1e-7);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
