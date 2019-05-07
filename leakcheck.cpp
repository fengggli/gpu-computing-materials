//
// Created by cmgoebel on 5/6/19.
//

#include "awnn/layer_conv.h"
#include "awnn/layer_pool.h"
#include "awnn/tensor.h"
#include "test_util.h"

#include "awnn/memory.h"

#include "awnndevice/layer_conv_device.cuh"
#include "awnndevice/memory.cuh"
#include <cstdio>
#include <numeric>
#include <vector>
#include <iostream>

using std::cout;

int main() {
  cublasHandle_t handle_;
  cublasStatus_t stat;
  stat = cublasCreate(&handle_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    PERR("CUBLAS initialization failed\n");
  }

  int nr_iterations = 1;
//  std::vector<int> block_arr = { 1, 2, 4, 8, 16, 32, 64 };
//  std::vector<int> thread_arr = { 2, 4, 8, 16, 32, 64, 128, 256, 512};
  std::vector<int> block_arr = { 64 };
  std::vector<int> thread_arr = { 256 };

  std::vector<int> N_arrary = {1, 4, 16}; // nr_imgs
  std::vector<int> C_arrary = {4}; // nr_input_channels
  std::vector<int> H_arrary = {32}; // input img sizes
  std::vector<int> F_arrary = {1, 4, 16}; // nr of filters/ output channels
  std::vector<int> HH_arrary = {3}; // filter H / W

  PINF("method\tnum_blk\tnum_thrd\tN\tC\tH\tF\tHH\tavg_fwd_ms\tavg_bkwd_ms");
  for (auto nr_img : N_arrary) {
    for (auto nr_in_channel : C_arrary) {
      for (auto sz_img : H_arrary) {
        for (auto nr_filter : F_arrary) {
          for (auto sz_filter : HH_arrary) {
            printf("\n");
            for (auto & num_blk : block_arr) {
              set_all_blocks(num_blk);
              for (auto & num_thrd : thread_arr) {
                set_all_threads(num_thrd);
                std::vector<double> forward_times;
                std::vector<double> backward_times;

                conv_param_t conv_params;

                conv_params.stride = 1;
                conv_params.padding = 1;
                int sz_out =
                    1 + (sz_img + 2 * conv_params.padding - sz_filter) /
                        conv_params.stride;
                // EXPECT_EQ(4, sz_out);

                int const shape_x[] = {nr_img, nr_in_channel, sz_img,
                                        sz_img};  // 2x3x4x4
                int const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                                        sz_filter};  // 3x3x3x3
                int const shape_y[] = {nr_img, nr_filter, sz_out,
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

                for (int i = 0; i < nr_iterations; i++) {
                auto t1 = get_timepoint();

                cout << "cache count before forward = " << cache.count << '\n';
                // FORWARD
                status_t ret = convolution_forward_device(
                    handle_, d_x, d_w, &cache, conv_params, d_y);
                assert(ret == S_OK);

                cout << "cache count after forward = " << cache.count << '\n';
                auto t2 = get_timepoint();
                forward_times.emplace_back(elapsed_ms(t1, t2));

                t1 = get_timepoint();

                ret = convolution_backward_device(handle_, d_dx, d_dw, &cache,
                                                  conv_params, d_dy);
                assert(ret == S_OK);

                cout << "cache count after backward = " << cache.count << '\n';
                t2 = get_timepoint();
                backward_times.emplace_back(elapsed_ms(t1, t2));
                }
                tensor_destroy(&x);
                tensor_destroy(&w);
                tensor_destroy(&y);
                tensor_destroy(&dy);
                tensor_destroy(&dx);
                tensor_destroy(&dw);

                tensor_destroy_device(d_x);
                tensor_destroy_device(d_w);
                tensor_destroy_device(d_y);

                tensor_destroy_device(d_dx);
                tensor_destroy_device(d_dw);
                tensor_destroy_device(d_dy);

                double avg_fwd_ms =
                    std::accumulate(forward_times.begin(), forward_times.end(),
                                    double(0)) /
                        (double)forward_times.size();
                double avg_bkwd_ms =
                    std::accumulate(backward_times.begin(),
                                    backward_times.end(), double(0)) /
                        (double)backward_times.size();

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
  cout << GET_TOTAL_TENSOR_ALLOC_DEVICE() -54 << '\n';
  cout << GET_TOTAL_TENSOR_DEALLOC_DEVICE() -54 << '\n';

}