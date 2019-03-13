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
  tensor_t t;
};

// Tests that the Foo::Bar() method does Abc.
TEST_F(TensorTest, Construct) {
  uint const shape[] = {3,4,5};
  t= tensor_make(shape, 3);
  EXPECT_TRUE(t.data != NULL);
}

/*TEST_F(TensorTest, Dump) {*/
// dim_t dim = make_dim(1, 3);
// EXPECT_EQ(dim.dims[0], 3);
// EXPECT_EQ(dim.dims[1], 0);
// EXPECT_EQ(dim.dims[2], 0);
// EXPECT_EQ(dim.dims[3], 0);
// EXPECT_EQ(dim_get_capacity(dim), 3);
// EXPECT_EQ(dim_get_ndims(dim), 1);
/*}*/

TEST_F(TensorTest, Destroy) {
  EXPECT_TRUE(t.data != NULL);
  tensor_destroy(t);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
