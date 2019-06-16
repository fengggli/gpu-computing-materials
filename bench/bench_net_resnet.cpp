/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "awnn/data_utils.h"
#include "awnn/layer_conv.h"
#include "awnn/net_resnet.h"
#include "awnn/solver.h"
#include "utils/data_cifar.h"
#include "utils/debug.h"
#include "utils/weight_init.h"

#include "gtest/gtest.h"
#include "test_util.h"
#include <numeric>
#undef PRINT_STAT

int main(int argc, char *argv[]) {
  static model_t model;
  // overfit small data;
  uint train_sz = 4000;
  // uint train_sz = 4000;
  uint batch_sz = 128;
  uint nr_iterations = 50;

  if(argc == 2){
    batch_sz = atoi(argv[1]);
  }

  uint input_shape[] = {batch_sz, 3, 32, 32};
  dim_t input_dim = make_dim_from_arr(array_size(input_shape), input_shape);
  uint output_dim = 10;
  uint nr_stages = 1;
  uint nr_blocks[MAX_STAGES] = {2};
  T reg = 1;
  normalize_method_t normalize_method = NORMALIZE_NONE;  // no batchnorm now

  // set_conv_method(CONV_METHOD_PERIMG);
  set_conv_method(CONV_METHOD_NNPACK_AUTO);
  resnet_init(&model, input_dim, output_dim, nr_stages, nr_blocks, reg,
              normalize_method);

  EXPECT_EQ((void *)0, (void *)net_get_param(model.list_all_params,
                                             "W3"));  // unexisting param
  EXPECT_NE((void *)0,
            (void *)net_get_param(model.list_all_params, "conv1.weight"));

  data_loader_t loader;
  status_t ret = cifar_open(&loader, CIFAR_PATH);
  EXPECT_EQ(S_OK, ret);

  uint val_sz = 1000;
  T learning_rate = 0.1;

  EXPECT_EQ(S_OK, cifar_split_train(&loader, train_sz, val_sz));

  tensor_t x;
  label_t *labels;

  // Read one batch of data
  uint cur_epoch = 0;
  uint cur_batch = 0;
  uint cnt_read = get_train_batch(&loader, &x, &labels, cur_batch, batch_sz);

  EXPECT_EQ(batch_sz, cnt_read);

  T loss = 0;
  resnet_loss(&model, x, labels, &loss);
  PINF("Initial Loss %.2f", loss);
  PINF("Using convolution method %d", get_conv_method());
  time_point_t start, end;
  std::vector<double> eclapsed_times;
  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    start = get_timepoint();
    resnet_loss(&model, x, labels, &loss);
    end = get_timepoint();
    double t = elapsed_ms(start, end);
    PINF("[iteration %u], forward-backward %.3fms", iteration, t);
    eclapsed_times.emplace_back(t);
  }
  double avg_ms =
      std::accumulate(eclapsed_times.begin(), eclapsed_times.end(), double(0)) /
      eclapsed_times.size();

  PINF("AVG forward-backward %.3fms", avg_ms);

  cifar_close(&loader);
  resnet_finalize(&model);
}

