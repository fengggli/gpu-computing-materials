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
class DimTest : public ::testing::Test {};

// Tests that the Foo::Bar() method does Abc.
TEST_F(DimTest, SingleDim) {
  dim_t dim = make_dim(1, 3);
  EXPECT_EQ(dim.dims[0], 3);
  EXPECT_EQ(dim.dims[1], 0);
  EXPECT_EQ(dim.dims[2], 0);
  EXPECT_EQ(dim.dims[3], 0);
  EXPECT_EQ(dim_get_capacity(dim), 3);
  EXPECT_EQ(dim_get_ndims(dim), 1);
  dim_dump(dim);
}

TEST_F(DimTest, ScaleDim) {
  dim_t dim = make_dim(0, 100);
  EXPECT_EQ(dim.dims[0], 0);
  EXPECT_EQ(dim.dims[1], 0);
  EXPECT_EQ(dim.dims[2], 0);
  EXPECT_EQ(dim.dims[3], 0);

  EXPECT_EQ(dim_get_capacity(dim), 1);
  EXPECT_EQ(dim_get_ndims(dim), 0);
  dim_dump(dim);
}

TEST_F(DimTest, FourDims) {
  dim_t dim = make_dim(4, 3, 4, 5, 6);
  EXPECT_EQ(dim.dims[0], 3);
  EXPECT_EQ(dim.dims[1], 4);
  EXPECT_EQ(dim.dims[2], 5);
  EXPECT_EQ(dim.dims[3], 6);

  EXPECT_EQ(dim_get_capacity(dim), 3 * 4 * 5 * 6);
  EXPECT_EQ(dim_get_ndims(dim), 4);
  dim_dump(dim);
}

TEST_F(DimTest, DimFromArray) {
  uint shape[] = {3, 4, 5, 6};
  dim_t dim = make_dim_from_arr(4, shape);
  EXPECT_EQ(dim.dims[0], 3);
  EXPECT_EQ(dim.dims[1], 4);
  EXPECT_EQ(dim.dims[2], 5);
  EXPECT_EQ(dim.dims[3], 6);

  EXPECT_EQ(dim_get_capacity(dim), 3 * 4 * 5 * 6);
  EXPECT_EQ(dim_get_ndims(dim), 4);
  dim_dump(dim);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
