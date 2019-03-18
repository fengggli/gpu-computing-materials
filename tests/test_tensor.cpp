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
  uint const shape[] = {2, 3, 1, 1}; // a 4-d tensor, but in memory this is the same as {2,3}
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

TEST_F(TensorTest, GetElem) {
  uint const shape[] = {2, 3, 4}; // a scalar
  tensor_t t1 = tensor_make_patterned(shape, dim_of_shape(shape));

  dim_t loc1 = make_dim(3,0,0,0);
  dim_t loc2 = make_dim(3,1,2,1);
  EXPECT_EQ(0, *tensor_get_elem_ptr(t1, loc1)); // todo location can be 0!
  EXPECT_EQ(21, *tensor_get_elem_ptr(t1, loc2));
  tensor_destroy(t1);
}

TEST_F(TensorTest, MakeTranspose) {
  uint const shape[] = {3, 4}; // a scalar
  tensor_t t1 = tensor_make_patterned(shape, dim_of_shape(shape));
  PINF("[-- dump t1]");
  tensor_dump(t1);

  tensor_t t2 = tensor_make_transpose(t1);
  PINF("[-- dump t2]");
  tensor_dump(t2);

  tensor_destroy(t1);
  tensor_destroy(t2);
}

TEST_F(TensorTest, MakeSum) {
  uint const shape[] = {3, 4}; // a scalar
  tensor_t t1 = tensor_make_patterned(shape, dim_of_shape(shape));
  uint axis_id = 0; // the axis along which sum will be performed.
  tensor_t t2 = tensor_make_sum(t1, axis_id);

  // t2 will have shape [1,4]
  EXPECT_EQ(1, t2.dim.dims[0]);
  EXPECT_EQ(4, t2.dim.dims[1]);

  EXPECT_EQ(12, t2.data[0]);
  EXPECT_EQ(15, t2.data[1]);
  EXPECT_EQ(18, t2.data[2]);
  EXPECT_EQ(21, t2.data[3]);

  tensor_destroy(t2);
  tensor_destroy(t1);
}

TEST_F(TensorTest, RelError) {
  uint const shape[] = {3, 4}; // a scalar
  tensor_t t1 = tensor_make_patterned(shape, dim_of_shape(shape));
  tensor_t t2 = tensor_make_copy(t1);

  T err_1 = tensor_rel_error(t1, t2);

  t2.data[0] += 0.001;
  T err_2 = tensor_rel_error(t1, t2);

  t2.data[0] += 0.001;
  T err_3 = tensor_rel_error(t1, t2);

  EXPECT_FLOAT_EQ(0.0, err_1);
  EXPECT_LT(err_1, err_2);
  EXPECT_LT(err_2, err_3);

  tensor_destroy(t2);
  tensor_destroy(t1);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
