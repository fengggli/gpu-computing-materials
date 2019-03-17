#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

#include "awnn/common.h"

namespace {

// The fixture for testing class Foo.
class TensorOpTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  TensorOpTest() { // You can do set-up work for each test here.
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
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 4};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));

  uint const shape3[] = {3, 4};
  out = tensor_make_patterned(shape3, dim_of_shape(shape3));

  EXPECT_TRUE(S_ERR == tensor_dot(in1, in2, out));
}

TEST_F(TensorOpTest, Dot) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));
  // tensor_dump(in1);

  uint const shape2[] = {3, 2};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));
  // tensor_dump(in2);

  uint const shape3[] = {2, 2};
  out = tensor_make_patterned(shape3, dim_of_shape(shape3));

  // int correct_result[] = {10, 13, 28, 40};

  EXPECT_EQ(S_OK, tensor_dot(in1, in2, out));
  // tensor_dump(out);
  EXPECT_EQ(out.data[0], 10);
  EXPECT_EQ(out.data[1], 13);
  EXPECT_EQ(out.data[2], 28);
  EXPECT_EQ(out.data[3], 40);
}

TEST_F(TensorOpTest, PLUS) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));
  // tensor_dump(in1);

  uint const shape2[] = {2, 3};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));
  // tensor_dump(in2);

  uint const shape3[] = {2, 3};
  out = tensor_make(shape3, dim_of_shape(shape3));

  EXPECT_EQ(S_OK, tensor_plus(in1, in2, out));
  EXPECT_EQ(out.data[0], 0);
  EXPECT_EQ(out.data[1], 2);
  EXPECT_EQ(out.data[2], 4);
  EXPECT_EQ(out.data[3], 6);
  EXPECT_EQ(out.data[4], 8);
  EXPECT_EQ(out.data[5], 10);
}

TEST_F(TensorOpTest, PLUS_INPLACE) {
  tensor_t from, to;
  uint const shape1[] = {2, 3};
  from = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 3};
  to = tensor_make_patterned(shape2, dim_of_shape(shape2));

  EXPECT_EQ(S_OK, tensor_plus_inplace(to, from));

  EXPECT_EQ(to.data[0], 0);
  EXPECT_EQ(to.data[1], 2);
  EXPECT_EQ(to.data[2], 4);
  EXPECT_EQ(to.data[3], 6);
  EXPECT_EQ(to.data[4], 8);
  EXPECT_EQ(to.data[5], 10);
}

TEST_F(TensorOpTest, RESHAPE) {
  tensor_t t;
  uint const shape1[] = {2, 3, 4};
  t = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 13}; // this shall gives a error
  uint const shape3[] = {2, 3, 2, 2};
  uint const shape4[] = {2, 12};
  uint const shape5[] = {24};

  EXPECT_EQ(S_BAD_DIM, tensor_reshape_(&t, shape2, 2));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape3, 4));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape4, 2));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape5, 1));

  EXPECT_EQ(t.dim.dims[0], 24);
  EXPECT_EQ(t.dim.dims[1], 0);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
