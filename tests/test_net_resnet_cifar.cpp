/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/data_utils.h"
#include "awnn/net_resnet.h"
#include "awnn/solver.h"
#include "utils/data_cifar.h"
#include "utils/debug.h"
#include "utils/weight_init.h"

#include "gtest/gtest.h"
#include "test_util.h"
#undef PRINT_STAT

namespace {

// The fixture for testing class Foo.
class NetMLPTest : public ::testing::Test {};

TEST_F(NetMLPTest, CifarTest) {
  static model_t model;
  uint batch_sz = 2;

  uint input_shape[] = {batch_sz, 3, 32, 32};
  dim_t input_dim = make_dim_from_arr(array_size(input_shape), input_shape);
  uint output_dim = 10;
  uint nr_stages = 1;
  uint nr_blocks[MAX_STAGES] = {2};
  T reg = 0;
  normalize_method_t normalize_method = NORMALIZE_NONE;  // no batchnorm now

  resnet_init(&model, input_dim, output_dim, nr_stages, nr_blocks, reg,
              normalize_method);

  EXPECT_EQ((void *)0, (void *)net_get_param(model.list_all_params,
                                             "W3"));  // unexisting param
  EXPECT_NE((void *)0,
            (void *)net_get_param(model.list_all_params, "fc0.weight"));

  data_loader_t loader;
  status_t ret = cifar_open(&loader, CIFAR_PATH);
  EXPECT_EQ(S_OK, ret);

  // overfit small data;
#ifdef IS_CI_BUILD  // make check faster
  uint train_sz = 2;
  uint nr_epoches = 1;
#else
  uint train_sz = 4000;
  uint nr_epoches = 5;
#endif

  uint val_sz = 1000;
  T learning_rate = 0.01;

  EXPECT_EQ(S_OK, cifar_split_train(&loader, train_sz, val_sz));

  uint iterations_per_epoch = train_sz / batch_sz;
  if (iterations_per_epoch == 0) iterations_per_epoch = 1;
  uint nr_iterations = nr_epoches * iterations_per_epoch;

  tensor_t x;
  label_t *labels;
  T loss = 0;

  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    uint cur_epoch = iteration / iterations_per_epoch;
    uint cur_batch = iteration % iterations_per_epoch;

    PINF("[Epoch %d, Iteration %u/%u]", cur_epoch, cur_batch,
         iterations_per_epoch);
    uint cnt_read = get_train_batch(&loader, &x, &labels, cur_batch, batch_sz);

    EXPECT_EQ(batch_sz, cnt_read);
    param_t *p_param;
#ifdef PRINT_STAT
    PINF("Before");
    // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
    list_for_each_entry(p_param, model.list_all_params, list) {
      tensor_t param = p_param->data;
      tensor_t dparam = p_param->diff;
      dump_tensor_stats(param, p_param->name);

      char diff_name[MAX_STR_LENGTH] = "";
      snprintf(diff_name, MAX_STR_LENGTH, "%s-diff", p_param->name);
      dump_tensor_stats(dparam, diff_name);
    }

#endif

    resnet_loss(&model, x, labels, &loss);

#ifdef PRINT_STAT
    PINF("After");
    // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
    list_for_each_entry(p_param, model.list_all_params, list) {
      tensor_t param = p_param->data;
      tensor_t dparam = p_param->diff;
      dump_tensor_stats(param, p_param->name);

      char diff_name[MAX_STR_LENGTH] = "";
      snprintf(diff_name, MAX_STR_LENGTH, "%s-diff", p_param->name);
      dump_tensor_stats(dparam, diff_name);
    }

#endif
    PINF("Loss %.2f", loss);

    // output the first/laster iteration, also in the end of each epoch
    if (iteration == 0 || iteration == nr_iterations - 1 ||
        cur_batch == iterations_per_epoch - 1) {
      PINF("----------------Epoch %u ---------------", cur_epoch);
      check_val_accuracy(&loader, val_sz, batch_sz, &model,
                         &resnet_forward_infer);
      check_train_accuracy(&loader, val_sz, batch_sz, &model,
                           &resnet_forward_infer);
      PINF("----------------------------------------");
    }

    // this will iterate fc0.weight, fc0.bias, fc1.weight, fc1.bias
    list_for_each_entry(p_param, model.list_all_params, list) {
      PDBG("updating %s...", p_param->name);
      // sgd
      sgd_update(p_param, learning_rate);
    }
  }
  cifar_close(&loader);
  resnet_finalize(&model);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
