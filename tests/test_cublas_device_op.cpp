/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */


#include "test_util.h"
#include "awnndevice/cublas_utils.cuh"

#include <gtest/gtest.h>

namespace {

// The fixture for testing class Foo.
  class cublasOpTests : public ::testing::Test {
  protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    cublasOpTests() {
      // You can do set-up work for each test here.
    }

    ~cublasOpTests() override {
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
//  TEST_F(cublasOpTests, transpose_2D_cublas_1) {
//    uint dim1 = 4;
//    uint dim2 = 2;
//
//    uint src_shape[] = { dim1, dim2 };
//    tensor_t src = tensor_make_device(src_shape, dim_of_shape(src_shape));
//
//    uint transpose_shape[] = { dim2, dim1 };
//    tensor_t tpose = tensor_make_device(transpose_shape, dim_of_shape(transpose_shape));
//
//    transpose_device(src);
//  }

TEST_F(cublasOpTests, transpose_mat_mult_cublas) {
  uint dim1 = 4;
  uint dim2 = 2;

  uint A_shape[] = { dim1, dim2 };
  tensor_t A = tensor_make_device(A_shape, dim_of_shape(A_shape));

  uint B_shape[] = { dim2, dim1 };
  tensor_t B = tensor_make_device(B_shape, dim_of_shape(B_shape));

  //////////////////////////////////////////
  tensor_t res = cublas_gemm_launch(A, B);
  //////////////////////////////////////////

  ASSERT_EQ(A_shape[0], res.dim.dims[0]);
  ASSERT_EQ(B_shape[1], res.dim.dims[1]);
}

#endif
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
