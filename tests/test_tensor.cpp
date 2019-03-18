/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "test_util.h"
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
  uint const shape[] = {2, 3, 4};;
  t = tensor_make(shape, dim_of_shape(shape));
  EXPECT_TRUE(t.data != nullptr);
}

TEST_F(TensorTest, Destroy) { tensor_destroy(t); }

TEST_F(TensorTest, Dumpd0) {

  uint const shape[] = {0}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd1) {
  uint const shape[] = {2}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd2) {
  uint const shape[] = {2, 3}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd3) {
  uint const shape[] = {2, 3, 4}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd4) {
  uint const shape[] = {2, 3, 4, 5}; // a scalar
  tensor_t tt = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, Dumpd4_2) {
  uint const shape[] = {2, 3, 1,1}; // a 4-d tensor, but in memory this is the same as {2,3}
  tensor_t tt = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_dump(tt);
  tensor_destroy(tt);
}

TEST_F(TensorTest, MakeLinspace) {
  uint const shape[] = {2, 2, 2, 2}; // a scalar
  tensor_t t1 = tensor_make_linspace(-0.1, 0.1, shape, dim_of_shape(shape));
  tensor_dump(t1);
  tensor_destroy(t1);
}

TEST_F(TensorTest, MakeCopy) {
  uint const shape[] = {2, 3, 4, 5}; // a scalar
  tensor_t t1 = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_t t2 = tensor_make_copy(t1);
  EXPECT_EQ(S_OK, dim_is_same(t1.dim, t2.dim));
  EXPECT_NE(t1.data, t2.data);
  tensor_destroy(t1);
  tensor_destroy(t2);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
