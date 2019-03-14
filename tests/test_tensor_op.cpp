#include "awnn/tensor.h"
#include "gtest/gtest.h"

#include "awnn/common.h"

namespace {

// The fixture for testing class Foo.
class TensorOpTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  TensorOpTest() {
    // You can do set-up work for each test here.
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

//
// tensor_t TensorOpTest::t;

// Tests that the Foo::Bar() method does Abc.
TEST_F(TensorOpTest, DotWrongInput) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, 2);

  uint const shape2[] = {2, 4};
  in2 = tensor_make_patterned(shape2, 2);

  uint const shape3[] = {3, 4};
  out = tensor_make_patterned(shape3, 2);

  EXPECT_TRUE(S_ERR == tensor_dot(in1, in2, out));
}

TEST_F(TensorOpTest, Dot) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, 2);
  tensor_dump(in1);

  uint const shape2[] = {3, 2};
  in2 = tensor_make_patterned(shape2, 2);
  tensor_dump(in2);

  uint const shape3[] = {2, 2};
  out = tensor_make_patterned(shape3, 2);

  int correct_result[] = {10, 13, 28, 40};

  EXPECT_TRUE(S_OK == tensor_dot(in1, in2, out));
  tensor_dump(out);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
