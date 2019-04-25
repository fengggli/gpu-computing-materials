/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

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
  T A_vals[] = { 1, 1,
                 1, 1,
                 1, 1,
                 1, 1 };
  tensor_t h_A = tensor_make(A_shape, dim_of_shape(A_shape));
  tensor_fill_list(h_A, A_vals, array_size(A_vals));

  uint B_shape[] = { dim2, dim1 };
  T B_vals[] = { 1, 1, 1, 1,
                 1, 1, 1, 1 };
  tensor_t h_B = tensor_make(B_shape, dim_of_shape(B_shape));
  tensor_fill_list(h_B, B_vals, array_size(B_vals));


  tensor_t d_A = tensor_make_copy_h2d(h_A);
  tensor_t d_B = tensor_make_copy_h2d(h_B);

  //////////////////////////////////////////
  tensor_t d_res = cublas_gemm_launch(handle_, d_A, d_B);
  //////////////////////////////////////////

  uint result_shape[] = { d_res.dim.dims[0], d_res.dim.dims[1] };
  tensor_t h_res = tensor_make(result_shape, dim_of_shape(result_shape));
  tensor_copy_d2h(h_res, d_res);

  ASSERT_EQ(A_shape[0], d_res.dim.dims[0]);
  ASSERT_EQ(B_shape[1], d_res.dim.dims[1]);

  for (int i = 0; i < 4; ++i) {
    std::cout << "\n  ";
    for (int j = 0; j < 4; ++j) {
//      std::cout << h_res.data[i * 4 + j] << ", ";
      ASSERT_EQ(2, h_res.data[i * 4 + j]);
    }
    std::cout << '\n';
  }

  tensor_destroy(&h_A);
  tensor_destroy(&h_B);
}

#endif
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
