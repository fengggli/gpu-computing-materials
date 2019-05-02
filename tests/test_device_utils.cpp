//
// Created by Christopher Goebel on 2019-05-02.
//

#include "test_util.h"
#include "awnn/common.h"

#include "awnndevice/cublas_wrappers.cuh"

#include <gtest/gtest.h>

namespace {

// The fixture for testing class Foo.
class cublasOpTests : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  cublasOpTests() {
    // You can do set-up work for each test here.
    stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      PERR("CUBLAS initialization failed\n");
    }
  }

  ~cublasOpTests() override {
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

TEST_F(deviceUtilTests, elementwise_add) {
  uint dim1 = 4;
  uint dim2 = 2;

  uint src_shape[] = { dim1, dim2 };
  tensor_t h_src = tensor_make_patterned(src_shape, dim_of_shape(src_shape));
  tensor_t h_out = tensor_make_alike(h_src);
  tensor_t d_a = tensor_make_copy_h2d(h_src);
  tensor_t d_b = tensor_make_copy_h2d(h_src);

  ////////////////////////////////////////////////////////
  elementwise_add_inplace_device(d_a, d_b); // sums into d_a
  ////////////////////////////////////////////////////////

  tensor_copy_d2h(h_out, d_a);

  for (int i = 0; i < tensor_get_capacity(h_src); ++i) {
      ASSERT_EQ(h_src.data[i] + h_src.data[i], h_out[i]);
  }

  tensor_destroy(&h_src);
  tensor_destroy(&h_out);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}

#endif // USE_CUDA


} // end namespace


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}