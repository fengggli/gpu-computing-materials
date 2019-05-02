//
// Created by Christopher Goebel on 2019-05-02.
//

#include "awnn/layer_conv.h"
#include "awnn/layer_pool.h"
#include "awnn/tensor.h"
#include "awnndevice/device_utils.cuh"
#include "awnndevice/layer_conv_device.cuh"
#include "test_util.h"

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
#include "awnn/memory.h"
#endif

#include <gtest/gtest.h>
#include <numeric>
#include <cstdio>

namespace {

using std::cout;

// The fixture for testing class Foo.
class TestDeviceUtil : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  TestDeviceUtil() {
    // You can do set-up work for each test here.
    stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      PERR("CUBLAS initialization failed\n");
    }
  }

  ~TestDeviceUtil() override {
    // You can do clean-up work that doesn't throw exceptions here.
    cublasDestroy(handle_);
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

  cublasHandle_t handle_;
  cublasStatus_t stat;
};


#ifdef USE_CUDA


TEST_F(TestDeviceUtil, elementwise_add_inplace_device) {
  uint dim1 = 4;
  uint dim2 = 2;

  uint src_shape[] = { dim1, dim2 };
  tensor_t h_src = tensor_make_patterned(src_shape, dim_of_shape(src_shape));
  tensor_t h_out = tensor_make_alike(h_src);
  tensor_t d_a = tensor_make_copy_h2d(h_src);
  tensor_t d_b = tensor_make_copy_h2d(h_src);

  ////////////////////////////////////////////////////////
  elementwise_add_inplace_device<<<1, 1>>>(d_a, d_b); // sums into d_a
  ////////////////////////////////////////////////////////

  tensor_copy_d2h(h_out, d_a);

  for (int i = 0; i < tensor_get_capacity(h_src); ++i) {
    ASSERT_EQ(h_src.data[i] + h_src.data[i], h_out.data[i]);
  }

  tensor_destroy(&h_src);
  tensor_destroy(&h_out);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}

#endif // USE_CUDA
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
