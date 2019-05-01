/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/layer_conv.h"
#include "awnn/layer_pool.h"
#include "awnn/tensor.h"
#include "awnndevice/layer_conv_device.cuh"
#include "test_util.h"

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
#include "awnn/memory.h"
#endif

#include <gtest/gtest.h>
#include <numeric>

namespace {

using std::cout;

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

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
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

  // input for backward
  tensor_t h_dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));
  tensor_t h_dx = tensor_make_alike(h_x);
  tensor_t h_dw = tensor_make_alike(h_w);

  long count_host_alloc   = GET_TOTAL_TENSOR_ALLOC_HOST();
  long count_device_alloc = GET_TOTAL_TENSOR_ALLOC_DEVICE();

  cout << "-----------------------------------------------------------------\n";
  cout << "BEGIN LOOP\n";

  int iterations = 450;
  for (int i = 0; i < iterations; ++i) {
    assert(cache.count == 0);

    reset_all_tensor_alloc_dealloc_stats();

    tensor_t d_y = tensor_make_copy_h2d(h_y);
    tensor_t d_x = tensor_make_copy_h2d(h_x);
    tensor_t d_w = tensor_make_copy_h2d(h_w);

    auto t1 = get_timepoint();
    /////////////////////////////////////////////////////////////////
    EXPECT_EQ(convolution_forward_device(handle_, d_x, d_w, &cache, params, d_y), S_OK);
    /////////////////////////////////////////////////////////////////
    auto t2 = get_timepoint();
    forward_times.emplace_back(elapsed_ms(t1, t2));

    tensor_destroy_device(&d_y);

    ASSERT_EQ(GET_TOTAL_TENSOR_ALLOC_DEVICE(), GET_TOTAL_TENSOR_DEALLOC_DEVICE() + cache.count);

    reset_all_tensor_alloc_dealloc_stats();

    tensor_t d_dy = tensor_make_copy_h2d(h_dy);
    tensor_t d_dx = tensor_make_copy_h2d(h_x);
    tensor_t d_dw = tensor_make_copy_h2d(h_w);

    t1 = get_timepoint();
    /////////////////////////////////////////////////////////////////
    EXPECT_EQ(convolution_backward_device(handle_, d_dx, d_dw, &cache, params, d_dy), S_OK);
    /////////////////////////////////////////////////////////////////
    t2 = get_timepoint();
    backward_times.emplace_back(elapsed_ms(t1, t2));

    tensor_destroy_device(&d_dw);
    tensor_destroy_device(&d_dx);
    tensor_destroy_device(&d_dy);

    ASSERT_EQ(GET_TOTAL_TENSOR_ALLOC_DEVICE() + cache.count, GET_TOTAL_TENSOR_DEALLOC_DEVICE());

    reset_all_tensor_alloc_dealloc_stats();
  }

  cout << "END LOOP\n";
  cout << "-----------------------------------------------------------------\n";

  double avg_fwd_ms = std::accumulate(forward_times.begin(), forward_times.end(), double(0)) / forward_times.size();
  double avg_bkwd_ms = std::accumulate(backward_times.begin(), backward_times.end(), double(0)) / backward_times.size();

  std::cout << "avg_fwd_ms=" << avg_fwd_ms << ", avg_bkwd_ms=" << avg_bkwd_ms << '\n';

  tensor_destroy(&h_x);
  tensor_destroy(&h_w);
  tensor_destroy(&h_y);
  tensor_destroy(&h_dy);
  tensor_destroy(&h_dx);
  tensor_destroy(&h_dw);


  print_memory_alloc_dealloc_totals();
  EXPECT_EQ(count_host_alloc, GET_TOTAL_TENSOR_DEALLOC_HOST());
  EXPECT_EQ(count_device_alloc, GET_TOTAL_TENSOR_DEALLOC_DEVICE());

}
#endif // #ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC


TEST_F(TestLayerConvSpeed, bench_custom_forward_backward) {
  uint nr_iterations = 100;
  std::vector<int> block_arr = { 1, 2, 4, 8, 16, 32, 64 };
  std::vector<int> thread_arr = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
  std::vector<uint> N_arrary = {1, 4, 16}; // nr_imgs
  std::vector<uint> C_arrary = {4}; // nr_input_channels
  std::vector<uint> H_arrary = {32}; // input img sizes
  std::vector<uint> F_arrary = {1, 4, 16}; // nr of filters/ output channels
  std::vector<uint> HH_arrary = {3}; // filter H / W

  PINF("method\tnum_blk\tnum_thrd\tN\tC\tH\tF\tHH\tavg_fwd_ms\tavg_bkwd_ms");
  for (auto & num_blk : block_arr) {
    set_blocks(num_blk);
    for (auto & num_thrd : thread_arr) {
      set_threads(num_thrd);
      for (auto nr_img : N_arrary) {
        for (auto nr_in_channel : C_arrary) {
          for (auto sz_img : H_arrary) {
            for (auto nr_filter : F_arrary) {
              for (auto sz_filter : HH_arrary) {
                std::vector<double> forward_times;
                std::vector<double> backward_times;

                conv_param_t conv_params;

                conv_params.stride = 1;
                conv_params.padding = 1;
                uint sz_out =
                    1 + (sz_img + 2 * conv_params.padding - sz_filter) /
                            conv_params.stride;
                // EXPECT_EQ(4, sz_out);

                uint const shape_x[] = {nr_img, nr_in_channel, sz_img,
                                        sz_img};  // 2x3x4x4
                uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                                        sz_filter};  // 3x3x3x3
                uint const shape_y[] = {nr_img, nr_filter, sz_out,
                                        sz_out};  // 2x3x4x4

                tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x,
                                                  dim_of_shape(shape_x));
                tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w,
                                                  dim_of_shape(shape_w));
                tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

                tensor_t d_x = tensor_make_copy_h2d(x);
                tensor_t d_w = tensor_make_copy_h2d(w);
                tensor_t d_y = tensor_make_copy_h2d(y);

                // input for backward
                tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y,
                                                   dim_of_shape(shape_y));

                tensor_t dx = tensor_make_alike(x);
                tensor_t dw = tensor_make_alike(w);

                tensor_t d_dy = tensor_make_copy_h2d(dy);
                tensor_t d_dx = tensor_make_copy_h2d(x);
                tensor_t d_dw = tensor_make_copy_h2d(w);

                lcache_t cache;
                make_empty_lcache(&cache);

                for (uint i = 0; i < nr_iterations; i++) {
                  auto t1 = get_timepoint();

                  // FORWARD
                  status_t ret = convolution_forward_device(
                      handle_, d_x, d_w, &cache, conv_params, d_y);
                  EXPECT_EQ(ret, S_OK);

                  auto t2 = get_timepoint();
                  forward_times.emplace_back(elapsed_ms(t1, t2));

                  t1 = get_timepoint();

                  ret = convolution_backward_device(handle_, d_dx, d_dw, &cache,
                                                    conv_params, d_dy);
                  EXPECT_EQ(ret, S_OK);

                  t2 = get_timepoint();
                  backward_times.emplace_back(elapsed_ms(t1, t2));
                }
                tensor_destroy(&x);
                tensor_destroy(&w);
                tensor_destroy(&y);
                tensor_destroy(&dy);
                tensor_destroy(&dx);
                tensor_destroy(&dw);

                tensor_destroy_device(&d_x);
                tensor_destroy_device(&d_w);
                tensor_destroy_device(&d_y);

                tensor_destroy_device(&d_dx);
                tensor_destroy_device(&d_dw);
                tensor_destroy_device(&d_dy);

                double avg_fwd_ms =
                    std::accumulate(forward_times.begin(), forward_times.end(),
                                    double(0)) /
                    forward_times.size();
                double avg_bkwd_ms =
                    std::accumulate(backward_times.begin(),
                                    backward_times.end(), double(0)) /
                    backward_times.size();

                PINF("stat-custom\t%i\t%i\t%u\t%u\t%u\t%u\t%u\t%.3f\t%.3f", 
                    num_blk, num_thrd, nr_img, nr_in_channel, sz_img, nr_filter,
                    sz_filter, avg_fwd_ms, avg_bkwd_ms);
              }
            }
          }
        }
      }
    }
  }
}

#endif // USE_CUDA
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
