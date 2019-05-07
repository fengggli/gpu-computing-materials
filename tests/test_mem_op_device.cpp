//
// Created by cmgoebel on 5/6/19.
//

#include "test_util.h"

#include "awnndevice/memory.cuh"
#include "awnndevice/tensor.cuh"

#include <gtest/gtest.h>

namespace {

// The fixture for testing class Foo.
class memOpDeviceTests : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  memOpDeviceTests() {}

  ~memOpDeviceTests() override {
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

};


#ifdef USE_CUDA
TEST_F(memOpDeviceTests, test_make_empty_device_tensor) {
  dim_t dim = { 1, 2, 3, 4 };

  tensor_t t = tensor_make_empty_device(dim);
  EXPECT_EQ(t.dim.dims[0], dim.dims[0]);
  EXPECT_EQ(t.dim.dims[1], dim.dims[1]);
  EXPECT_EQ(t.dim.dims[2], dim.dims[2]);
  EXPECT_EQ(t.dim.dims[3], dim.dims[3]);

  EXPECT_EQ(t.data, nullptr);
}

TEST_F(memOpDeviceTests, full_tensor_make_device) {

  int iterations = 100;
  for (int i = 0; i < iterations; ++i) {
    int shape[] = { 3, 2, 4, 5 };
    tensor_t t = tensor_make_device(shape, dim_of_shape(shape));
    EXPECT_EQ(t.allocation_tag, i);

    EXPECT_EQ(t.dim.dims[0], shape[0]);
    EXPECT_EQ(t.dim.dims[1], shape[1]);
    EXPECT_EQ(t.dim.dims[2], shape[2]);
    EXPECT_EQ(t.dim.dims[3], shape[3]);

    EXPECT_EQ(tensor_get_capacity(t), shape[1] * shape[2] * shape[3] * shape[4]);

    EXPECT_NE(t.data, nullptr);

    tensor_destroy_device(&t);

    // deallocations should result in nullptr
    EXPECT_EQ(t.data, nullptr);

    // deallocations do not destroy shape
    EXPECT_EQ(t.dim.dims[0], shape[0]);
    EXPECT_EQ(t.dim.dims[1], shape[1]);
    EXPECT_EQ(t.dim.dims[2], shape[2]);
    EXPECT_EQ(t.dim.dims[3], shape[3]);

    // dealloc count must match alloc count
    EXPECT_EQ(GET_TOTAL_TENSOR_ALLOC_DEVICE(), GET_TOTAL_TENSOR_DEALLOC_DEVICE());
  }
}

#endif // USE_CUDA
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
