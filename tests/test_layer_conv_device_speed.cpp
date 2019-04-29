/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */


#include "test_util.h"
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "awnn/layer_pool.h"
#include "awnndevice/dev_layer_conv.cuh"

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
#include "awnn/memory.h"
#endif

#include <gtest/gtest.h>
#include <numeric>

namespace {

// The fixture for testing class Foo.
  class TestLayerConvSpeed : public ::testing::Test {
  protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    TestLayerConvSpeed() {
      // You can do set-up work for each test here.
      stat = cublasCreate(&handle_);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        PERR("CUBLAS initialization failed\n");
      }
    }

    ~TestLayerConvSpeed() override {
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

TEST_F(TestLayerConvSpeed, forward_and_backward_loop) {
  std::vector<double> forward_times;
  std::vector<double> backward_times;

  lcache_t cache;
  make_empty_lcache(&cache);

  conv_param_t params = { .stride = 2, .padding = 1 };

  uint nr_img = 2;
  uint sz_img = 4;
  uint nr_in_channel = 3;
  uint sz_filter = 4;
  uint nr_filter = 3;

  uint sz_out = 1 + (sz_img + 2 * params.padding - sz_filter) / params.stride;
  EXPECT_EQ(2, sz_out);

  uint const shape_x[] = { nr_img, nr_in_channel, sz_img, sz_img }; // 2x3x4x4
  uint const shape_w[] = { nr_filter, nr_in_channel, sz_filter, sz_filter }; // 3x3x4x4
  uint const shape_y[] = { nr_img, nr_filter, sz_out, sz_out }; // 2x3x2x2

  tensor_t h_x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t h_w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t h_y = tensor_make(shape_y, dim_of_shape(shape_y));

  tensor_t d_x = tensor_make_copy_h2d(h_x);
  tensor_t d_w = tensor_make_copy_h2d(h_w);
  tensor_t d_y = tensor_make_copy_h2d(h_y);

  // input for backward
  tensor_t h_dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));
  tensor_t d_dy = tensor_make_copy_h2d(h_dy);
  tensor_t h_dx = tensor_make_alike(h_x);
  tensor_t d_dx = tensor_make_copy_h2d(h_x);
  tensor_t h_dw = tensor_make_alike(h_w);
  tensor_t d_dw = tensor_make_copy_h2d(h_w);


  int iterations = 10;
  for (int i = 0; i < iterations; ++i) {
    std::cout << "cache_size before forward = " << cache.count << '\n';

    auto t1 = get_timepoint();
    /////////////////////////////////////////////////////////////////
    EXPECT_EQ(convolution_forward_device(handle_, d_x, d_w, &cache, params, d_y), S_OK);
    /////////////////////////////////////////////////////////////////
    auto t2 = get_timepoint();
    forward_times.emplace_back(elapsed_ms(t1, t2));

    std::cout << "cache_size after forward / before backward = " << cache.count << '\n';

    t1 = get_timepoint();
    /////////////////////////////////////////////////////////////////
    EXPECT_EQ(convolution_backward_device(handle_, d_dx, d_dw, &cache, params, d_dy), S_OK);
    /////////////////////////////////////////////////////////////////
    t2 = get_timepoint();
    backward_times.emplace_back(elapsed_ms(t1, t2));

    std::cout << "cache_size after backward = " << cache.count << '\n';
  }
  double avg_fwd_ms = std::accumulate(forward_times.begin(), forward_times.end(), double(0)) / forward_times.size();
  double avg_bkwd_ms = std::accumulate(backward_times.begin(), backward_times.end(), double(0)) / backward_times.size();

  std::cout << "avg_fwd_ms=" << avg_fwd_ms << ", avg_bkwd_ms=" << avg_bkwd_ms << '\n';

  tensor_destroy_device(&d_dw);
  tensor_destroy_device(&d_dx);
  tensor_destroy(&h_dy);
  tensor_destroy(&h_dx);
  tensor_destroy(&h_dw);

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC

  std::cout << "total host allocations = "
            << GET_TOTAL_TENSOR_ALLOC_HOST()
            << ", total host de-allocations = "
            << GET_TOTAL_TENSOR_DEALLOC_HOST() << '\n';
  EXPECT_EQ(GET_TOTAL_TENSOR_ALLOC_HOST(), GET_TOTAL_TENSOR_DEALLOC_HOST());
  std::cout << "total device allocations = "
            << GET_TOTAL_TENSOR_ALLOC_DEVICE()
            << ", total device de-allocations = "
            << GET_TOTAL_TENSOR_DEALLOC_DEVICE() << '\n';

  EXPECT_EQ(GET_TOTAL_TENSOR_ALLOC_DEVICE(), GET_TOTAL_TENSOR_DEALLOC_DEVICE());
#endif
}

#endif
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
