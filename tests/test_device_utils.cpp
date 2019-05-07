//
// Created by Christopher Goebel on 2019-05-02.
//

#include "awnn/layer_conv.h"
#include "awnn/layer_pool.h"
#include "awnn/tensor.h"
#include "awnndevice/device_utils_harness.cuh"
#include "test_util.h"

#include <cublas_v2.h>

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
    set_build_mask_blocks(32);
    set_build_mask_threads(128);
    set_elementwise_add_blocks(32);
    set_elementwise_add_threads(128);
    set_elementwise_mul_blocks(32);
    set_elementwise_mul_threads(128);
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


TEST_F(TestDeviceUtil, elementwise_add_host_harness) {
  int dim1 = 4;
  int dim2 = 2;

  int src_shape[] = { dim1, dim2 };
  tensor_t h_src = tensor_make_patterned(src_shape, dim_of_shape(src_shape));
  tensor_t h_out = tensor_make_copy(h_src);

  ////////////////////////////////////////////////////////
  elementwise_add_device_host_harness(h_out, h_src); // sums into d_a
  ///////////////////////////////////////////////////////

  for (int i = 0; i < tensor_get_capacity(h_src); ++i) {
    EXPECT_FLOAT_EQ(float(h_src.data[i] + h_src.data[i]), float(h_out.data[i]));
  }

  tensor_destroy(&h_src);
  tensor_destroy(&h_out);
}

TEST_F(TestDeviceUtil, elementwise_add_device_harness) {
  int dim1 = 4;
  int dim2 = 2;

  int src_shape[] = { dim1, dim2 };
  tensor_t h_b = tensor_make_patterned(src_shape, dim_of_shape(src_shape));
  tensor_t h_a = tensor_make_copy(h_b);

  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  ////////////////////////////////////////////////////////
  elementwise_add_device_harness(d_a, d_b); // sums into d_a
  ///////////////////////////////////////////////////////

  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);

  for (int i = 0; i < tensor_get_capacity(h_b); ++i) {
    EXPECT_FLOAT_EQ(float(h_b.data[i] + h_b.data[i]), float(h_a.data[i]));
  }

  tensor_destroy(&h_b);
  tensor_destroy(&h_a);
}

TEST_F(TestDeviceUtil, elementwise_mul_host_harness) {
  int dim1 = 4;
  int dim2 = 2;

  int src_shape[] = { dim1, dim2 };
  tensor_t h_src = tensor_make_patterned(src_shape, dim_of_shape(src_shape));
  tensor_t h_out = tensor_make_copy(h_src);

  ////////////////////////////////////////////////////////
  elementwise_mul_device_host_harness(h_out, h_src); // muls into d_a
  ///////////////////////////////////////////////////////

  for (int i = 0; i < tensor_get_capacity(h_src); ++i) {
    EXPECT_FLOAT_EQ(float(h_src.data[i] * h_src.data[i]), float(h_out.data[i]));
  }

  tensor_destroy(&h_src);
  tensor_destroy(&h_out);
}


TEST_F(TestDeviceUtil, elementwise_mul_device_harness) {
  int dim1 = 4;
  int dim2 = 2;

  int src_shape[] = { dim1, dim2 };
  tensor_t h_b = tensor_make_patterned(src_shape, dim_of_shape(src_shape));
  tensor_t h_a = tensor_make_copy(h_b);

  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  ////////////////////////////////////////////////////////
  elementwise_mul_device_harness(d_a, d_b); // muls into d_a
  ///////////////////////////////////////////////////////

  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);

  for (int i = 0; i < tensor_get_capacity(h_b); ++i) {
    EXPECT_FLOAT_EQ(float(h_b.data[i] * h_b.data[i]), float(h_a.data[i]));
  }

  tensor_destroy(&h_b);
  tensor_destroy(&h_a);
}




TEST_F(TestDeviceUtil, apply_mask_host_harness) {
  int dim1 = 4;
  int dim2 = 2;

  int src_shape[] = { dim1, dim2 };
  tensor_t h_a = tensor_make(src_shape, dim_of_shape(src_shape));
  tensor_t h_mask = tensor_make_zeros_alike(h_a);

  for(int i = 0; i < dim1 * dim2; ++i) {
    if(i % 2 == 0) {
      h_a.data[i] = 33;
    } else {
      h_a.data[i] = 0;
    }
  }

  ////////////////////////////////////////////////////////
  build_mask_device_host_harness(h_a, h_mask); // sums into d_a
  ///////////////////////////////////////////////////////

  for (int i = 0; i < tensor_get_capacity(h_a); ++i) {
    if(i % 2 == 0) {
      EXPECT_FLOAT_EQ(1, h_mask.data[i]);
    }
  }

  tensor_destroy(&h_a);
  tensor_destroy(&h_mask);
}


TEST_F(TestDeviceUtil, apply_mask_device_harness) {
  int dim1 = 4;
  int dim2 = 2;

  int src_shape[] = { dim1, dim2 };
  tensor_t h_a = tensor_make(src_shape, dim_of_shape(src_shape));
  tensor_t h_mask = tensor_make_zeros_alike(h_a);

  for(int i = 0; i < dim1 * dim2; ++i) {
    if(i % 2 == 0) {
      h_a.data[i] = 33;
    } else {
      h_a.data[i] = 0;
    }
  }

  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_mask = tensor_make_copy_h2d(h_mask);

  ////////////////////////////////////////////////////////
  build_mask_device_harness(d_a, d_mask);
  ///////////////////////////////////////////////////////

  tensor_copy_d2h(h_mask, d_mask);

  tensor_destroy_device(&d_mask);
  tensor_destroy_device(&d_a);

  for (int i = 0; i < tensor_get_capacity(h_a); ++i) {
    if(i % 2 == 0) {
      EXPECT_FLOAT_EQ(1, h_mask.data[i]);
    }
  }

  tensor_destroy(&h_a);
  tensor_destroy(&h_mask);
}

#endif // USE_CUDA
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
