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
#include "layers/layer_common.hpp"
#include "utils/data_cifar.h"
#include "utils/debug.h"
#include "utils/weight_init.h"

#include "gtest/gtest.h"
#include "test_util.h"
#include <numeric>
#undef PRINT_STAT

/** work thead entry*/

int main(int argc, char *argv[]) {
  // overfit small data;
  uint train_sz = 4000;
  // uint train_sz = 4000;
  int batch_sz = 128;

  if(argc != 4){
    PERR("format: cmd batch_size nr_worker_threads nr_iterations");
    return -1;
  }

  batch_sz = atoi(argv[1]);
  int nr_threads = atoi(argv[2]);
  int nr_iterations = atoi(argv[3]);

  resnet_main(batch_sz, nr_threads, nr_iterations);
}

