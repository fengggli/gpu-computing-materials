/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include <vector>
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

#include "awnn/layer_cudnn.h"

namespace {
class LayerBenchConvDeviceTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerBenchConvDeviceTest, BenchCUDNN) {
  uint nr_iterations = 100;
  std::vector<uint> N_arrary = {1, 4, 16}; // nr_imgs
  std::vector<uint> C_arrary = {4}; // nr_input_channels
  std::vector<uint> H_arrary = {32}; // input img sizes
  std::vector<uint> F_arrary = {1, 4, 16}; // nr of filters/ output channels
  std::vector<uint> HH_arrary = {3}; // filter H / W

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
            uint sz_out = 1 + (sz_img + 2 * conv_params.padding - sz_filter) /
                                  conv_params.stride;
            // EXPECT_EQ(4, sz_out);

            uint const shape_x[] = {nr_img, nr_in_channel, sz_img,
                                    sz_img};  // 2x3x4x4
            uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                                    sz_filter};  // 3x3x3x3
            uint const shape_y[] = {nr_img, nr_filter, sz_out,
                                    sz_out};  // 2x3x4x4

            tensor_t x =
                tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
            tensor_t w =
                tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
            tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

            tensor_t d_x = tensor_make_copy_h2d(x);
            tensor_t d_w = tensor_make_copy_h2d(w);
            tensor_t d_y = tensor_make_copy_h2d(y);

            // input for backward
            tensor_t dy =
                tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));

            tensor_t dx = tensor_make_alike(x);
            tensor_t dw = tensor_make_alike(w);

            tensor_t d_dy = tensor_make_copy_h2d(dy);
            tensor_t d_dx = tensor_make_copy_h2d(x);
            tensor_t d_dw = tensor_make_copy_h2d(w);

            lcache_t cache;
            make_empty_lcache(&cache);

            cudnnHandle_t handle_;
            cudnnTensorDescriptor_t cudnnIdesc;
            cudnnFilterDescriptor_t cudnnFdesc;
            cudnnTensorDescriptor_t cudnnOdesc;
            cudnnConvolutionDescriptor_t cudnnConvDesc;

            checkCudnnErr(cudnnCreate(&handle_));

            checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
            checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
            checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
            checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

            for (uint i = 0; i < nr_iterations; i++) {
              auto t1 = get_timepoint();

              // FORWARD
              status_t ret =
                  convolution_forward_cudnn(d_x, d_w, &cache, conv_params, d_y,
                      handle_, cudnnIdesc,  cudnnFdesc, cudnnOdesc, cudnnConvDesc);
              EXPECT_EQ(ret, S_OK);

              auto t2 = get_timepoint();
              forward_times.emplace_back(elapsed_ms(t1, t2));

              t1 = get_timepoint();

              ret = convolution_backward_cudnn(d_dx, d_dw, &cache, conv_params, d_dy);
              EXPECT_EQ(ret, S_OK);

              t2 = get_timepoint();
              backward_times.emplace_back(elapsed_ms(t1, t2));
            }

            clean:
            if (cudnnIdesc) cudnnDestroyTensorDescriptor(cudnnIdesc);
            if (cudnnFdesc) cudnnDestroyFilterDescriptor(cudnnFdesc);
            if (cudnnOdesc) cudnnDestroyTensorDescriptor(cudnnOdesc);
            if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
            if (handle_) cudnnDestroy(handle_);

            tensor_destroy(&x);
            tensor_destroy(&w);
            tensor_destroy(&y);
            tensor_destroy(&dy);
            tensor_destroy(&dx);
            tensor_destroy(&dw);
            double avg_fwd_ms =
                std::accumulate(forward_times.begin(), forward_times.end(),
                                double(0)) /
                forward_times.size();
            double avg_bkwd_ms =
                std::accumulate(backward_times.begin(), backward_times.end(),
                                double(0)) /
                backward_times.size();

            PINF("[stat]");
            PINF("method\tN\tC\tH\tF\tHH\tavg_fwd_ms\tavg_bkwd_ms");
            PINF("stat-cudnn\t%u\t%u\t%u\t%u\t%u\t%.3f\t%.3f", nr_img, nr_in_channel,
                 sz_img, nr_filter, sz_filter, avg_fwd_ms, avg_bkwd_ms);
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
