/** Bench covolution layers*/
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "mkl.h"

struct {
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
  uint sz_img = 32;
  uint nr_in_channel = 16;
  uint sz_filter = 3;
  uint nr_filter = 16;

  uint sz_out =
      1 + (sz_img + 2 * conv_params.padding - sz_filter) / conv_params.stride;
  EXPECT_EQ(32, sz_out);

  uint const shape_x[] = {nr_img, nr_in_channel, sz_img, sz_img};  // 2x3x4x4
  uint const shape_w[] = {nr_filter, nr_in_channel, sz_filter,
                          sz_filter};                          // 3x3x3x3
  uint const shape_y[] = {nr_img, nr_filter, sz_out, sz_out};  // 2x3x4x4

	// this take prcedence over openmp_set_num_threads
	mkl_set_dynamic(0);// always use provided number
  mkl_set_num_threads(nr_blas_threads);
	PMAJOR("setting mkl thread to %d", nr_blas_threads);

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

int main(int argc, char **argv) {
	if (argc != 2) {
    PWRN("format: bench_conv threadnumber");
    return 0;
  }
	opt.nr_blas_threads = atoi(argv[1]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
