/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "test_util.h"
#include "awnn/layer_conv.h"
#include "awnn/tensor.h"

#include "gtest/gtest.h"

namespace {

// The fixture for testing class Foo.
  class LayerConvTestDevice : public ::testing::Test {
  protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    LayerConvTestDevice() {
      // You can do set-up work for each test here.
    }

    ~LayerConvTestDevice() override {
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


// TODO: check with cudnn
#ifdef USE_CUDA
  TEST_F(LayerConvTestDevice, forward_device_01
  ) {
  conv_param_t conv_params = {1, 0};

  uint n = 1;
  uint img_sz = 3;
  uint c = 2;
  uint fltr_sz = 2;
  uint num_fil = 2;
  uint sz_out = 1 + (img_sz + 2 * conv_params.padding - fltr_sz) / conv_params.stride;

  uint const shape_x[] = {n, c, img_sz, img_sz}; // 2x3x4x4
  uint const shape_w[] = {num_fil, c, fltr_sz, fltr_sz}; // 3x3x4x4
  uint const shape_y[] = {n, num_fil, sz_out, sz_out};

  EXPECT_EQ(2, sz_out);

  T x_values[] = {1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2};
  tensor_t x = tensor_make(shape_x, dim_of_shape(shape_x));
  tensor_fill_list(x, x_values, array_size(x_values)
  );

  T w_values[] = {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2};
  tensor_t w = tensor_make(shape_w, dim_of_shape(shape_w));
  tensor_fill_list(w, w_values, array_size(w_values)
  );

  tensor_t y = tensor_make(shape_y, dim_of_shape(shape_y));
  lcache_t cache;
  make_empty_lcache(&cache);

  status_t ret = convolution_forward_device(x, w, &cache, conv_params, y);
  EXPECT_EQ(ret, S_OK
  );

  tensor_t y_ref = tensor_make_alike(y);
  T value_list[] = {6, 2, 3, 4, 3, 3, 7, 7};
  tensor_fill_list(y_ref, value_list, array_size(value_list)
  );

  EXPECT_LT(tensor_rel_error(y_ref, y),
  1e-7);
  PINF("Consistent with expected results");
}
#endif

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
