/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/tensor.h"
#include "gtest/gtest.h"

namespace {

// The fixture for testing class Foo.
class TensorTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  TensorTest() {
    // You can do set-up work for each test here.
  }

  ~TensorTest() override {
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
  static tensor_t t;
};

//
tensor_t TensorTest::t;

// Tests that the Foo::Bar() method does Abc.
TEST_F(TensorTest, Construct) {
  uint const shape[] = {2, 3, 4};
  t= tensor_make(shape, 3);
  EXPECT_TRUE(t.data != NULL);
}

TEST_F(TensorTest, Destroy) { tensor_destroy(t); }

TEST_F(TensorTest, Dumpd0) {
  uint const shape[] = {0}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, 0);
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd1) {
  uint const shape[] = {2}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, 1);
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd2) {
  uint const shape[] = {2, 3}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, 2);
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd3) {
  uint const shape[] = {2, 3, 4}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, 3);
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd4) {
  uint const shape[] = {2, 3, 4, 5}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, 4);
  tensor_dump(tt);
  tensor_destroy(tt);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
