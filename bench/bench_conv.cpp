/** Bench covolution layers*/
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "mkl.h"

struct {
	int imsize = 32;
  int nr_blas_threads = 4;
  int nr_iters = 5000;
}opt;

namespace {
class LayerConvPerImgTest : public ::testing::Test {};
}  // namespace

TEST_F(LayerConvPerImgTest, Forward) {
  int nr_blas_threads = opt.nr_blas_threads;
  int i, nr_iters = opt.nr_iters;
  conv_param_t conv_params;

  conv_params.stride = 1;
  conv_params.padding = 1;

  uint nr_img = 1;
  uint sz_img = opt.imsize;
  uint nr_in_channel = 16;
  uint sz_filter = 3;
  uint nr_filter = 16;

  uint sz_out =
      1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x3x3
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x4x4

	// this take prcedence over openmp_set_num_threads
	mkl_set_dynamic(0);// always use provided number
  mkl_set_num_threads(nr_blas_threads);
	PMAJOR("setting mkl thread to %d, imsize= %d", nr_blas_threads, sz_img);

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  set_conv_method(CONV_METHOD_PERIMG);
  PINF("Forward Using method %d", get_conv_method());

  time_point_t start, end;
  std::vector<double> eclapsed_times;

  start = get_timepoint();

  for(i = 0; i < nr_iters; i++){
    status_t ret = convolution_forward(
        x, w, NULL, conv_params,
        y);  // forward function should allocate and populate cache;
     EXPECT_EQ(ret, S_OK);
    }
    end = get_timepoint();
    double t = elapsed_ms(start, end);
    PINF("forward-backward %.3fms", t);
}

TEST_F(LayerConvPerImgTest, DISABLED_Backward) {
  conv_param_t conv_params;

  conv_params.stride = 1;
  conv_params.padding = 1;

  uint nr_img = 1;
  uint sz_img = opt.imsize;
  uint nr_in_channel = 16;
  uint sz_filter = 3;
  uint nr_filter = 16;

  uint sz_out =
      1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x3x3
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x4x4

	mkl_set_dynamic(0);// always use provided number
  mkl_set_num_threads(opt.nr_blas_threads);
	PMAJOR("setting mkl thread to %d, imsize= %d", opt.nr_blas_threads, sz_img);

  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));
  tensor_t w = tensor_make_linspace(-0.2, 0.3, shape_w, dim_of_shape(shape_w));
  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));

  lcache_t cache;
  make_empty_lcache(&cache);

  set_conv_method(CONV_METHOD_PERIMG);
  status_t ret = convolution_forward(
      x, w, &cache, conv_params,
      y);  // forward function should allocate and populate cache;
  EXPECT_EQ(ret, S_OK);

  // input for backward
  tensor_t dy = tensor_make_linspace(-0.1, 0.5, shape_y, dim_of_shape(shape_y));

  tensor_t dx = tensor_make_alike(x);
  tensor_t dw = tensor_make_alike(w);

  PINF("Backward Using method %d", get_conv_method());

  time_point_t start, end;
  std::vector<double> eclapsed_times;

  start = get_timepoint();

  for(int i = 0; i < opt.nr_iters; i++){

  		ret = convolution_backward(dx, dw, &cache, conv_params,
                             dy);  // backward needs to call free_lcache(cache);
     EXPECT_EQ(ret, S_OK);
  }
    end = get_timepoint();
    double t = elapsed_ms(start, end);
    PINF("Back-backward %.3fms", t);

  EXPECT_EQ(ret, S_OK);

  /* II. Numerical check */
  // I had to make this copy since lambda doesn't allow me to use global
  // variable
  tensor_t x_copy = tensor_make_copy(x);
  tensor_t w_copy = tensor_make_copy(w);

  tensor_t dx_ref = tensor_make_alike(x);
  tensor_t dw_ref = tensor_make_alike(w);

  // evaluate gradient of x
  eval_numerical_gradient(
      [&](tensor_t const in, tensor_t out) {
        convolution_forward(in, w_copy, nullptr, conv_params, out);
      },
      x, dy, dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-4);
  PINF("gradient check of x... is ok");

  // evaluate gradient of w
  eval_numerical_gradient(
      [&](tensor_t const in, tensor_t out) {
        convolution_forward(x_copy, in, nullptr, conv_params, out);
      },
      w, dy, dw_ref);
  EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-4);
  PINF("gradient check of w... is ok");

  EXPECT_EQ(ret, S_OK);
}

int main(int argc, char **argv) {
	if (argc != 3) {
    PWRN("format: bench_conv imsize threadnumber");
    return 0;
  }
	opt.imsize = atoi(argv[1]);
	opt.nr_blas_threads = atoi(argv[2]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
