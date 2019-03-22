/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/loss_softmax.h"
#include "awnn/tensor.h"
#include "test_util.h"
#include "gtest/gtest.h"
#include <cmath>

namespace {

// The fixture for testing class Foo.
class LostSoftmaxTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  LostSoftmaxTest() {
    // You can do set-up work for each test here.
  }

  ~LostSoftmaxTest() override {
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

  // Objects declared here can be used by all tests.
};

TEST_F(LostSoftmaxTest, Loss) {
  uint nr_images = 3073;
  uint nr_classes = 10;
}

TEST_F(LostSoftmaxTest, OneImg) {

  uint nr_classes = 3;
  uint nr_images = 2;

  uint const shape_x[] = {
      nr_images, nr_classes}; // e.g. nr_images images, nr_classes features (softmax usually
                              // follows fc, which is already flattened to 2d)
  tensor_t x = tensor_make(shape_x, dim_of_shape(shape_x));
  T const value_list[] = {-2.85, 0.86, 0.28, -2.85, 0.86, 0.28};
  //T const value_list[] = {-2.85, 0.86, -2.85, 0.86};
  tensor_fill_list(x, value_list, nr_images*nr_classes);

  status_t ret;
  label_t real_labels[] = {1, 1}; // change to 2,2 will trigger an err

  T loss;
  tensor_t dx = tensor_make_alike(x);

  ret = loss_softmax(x, real_labels, &loss, MODE_TRAIN, dx);
  //EXPECT_TRUE(fabs(1.04 - loss) <
  //            1e-3); // http://cs231n.github.io/linear-classify/

  PINF("softmax loss of %d images is: %.3f", nr_images, loss);
  EXPECT_EQ(ret, S_OK);

  PINF("backward gradient:");
  tensor_dump(dx);

  auto func_softmax = [real_labels, dx](tensor_t const input, tensor_t output) {
    T ref_loss;
    loss_softmax(input, real_labels, &ref_loss, MODE_INFER, dx);
    output.data[0] = ref_loss;
  };

  uint const unit_shape[] = {1};
  tensor_t unit_t = tensor_make_ones(unit_shape, dim_of_shape(unit_shape));
  tensor_t dx_ref = tensor_make_alike(x);
  eval_numerical_gradient(func_softmax, x, unit_t, dx_ref, 1e-3);

  PINF("input: (modified (+/-h) data)");
  tensor_dump(x);

  PINF("output (gradient of input):");
  tensor_dump(dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-5);
  PINF("gradient check of x... is ok");

  tensor_destroy(dx_ref);
}

// see whether gradient for softmax is generated correctly for multiple images
TEST_F(LostSoftmaxTest, DISABLED_MultiImg) {

  uint nr_images = 6;
  uint nr_classes = 10;

  uint const shape_x[] = {
      nr_images, nr_classes}; // e.g. 3 images, 4 features (softmax usually
                              // follows fc, which is already flattened to 2d)
  tensor_t x = tensor_make_linspace(-0.1, 0.5, shape_x, dim_of_shape(shape_x));

  label_t real_labels[] = {2, 3, 4, 5, 1, 2};

  tensor_t dx =
      tensor_make_alike(x); // this is not actually required for inference

  auto func_softmax = [real_labels, dx](tensor_t const input, tensor_t output) {
    T ref_loss;
    loss_softmax(input, real_labels, &ref_loss, MODE_INFER, dx);
    output.data[0] = ref_loss;
  };

  uint const unit_shape[] = {1};
  tensor_t unit_t = tensor_make(
      unit_shape,
      dim_of_shape(
          unit_shape)); // softmax is last layer, no gradient from above
  tensor_t dx_ref = tensor_make_alike(x);
  unit_t.data[0] = 1.0;
  //eval_numerical_gradient(func_softmax, x, unit_t, dx_ref, 1e-5);

  PINF("backward gradient:");
  tensor_dump(dx);
  PINF("nuercial gradient:");
  tensor_dump(dx_ref);
  EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
  PINF("gradient check of x... is ok");

  tensor_destroy(dx_ref);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
