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

#include <numeric>
#include "gtest/gtest.h"
#include "test_util.h"
#undef PRINT_STAT

/** work thead entry*/

int main(int argc, char *argv[]) {
  // overfit small data;
  uint train_sz = 4000;
  // uint train_sz = 4000;
  int batch_sz = 128;

  if (argc != 5) {
    PERR("format: cmd modelname batch_size nr_worker_threads nr_iterations\n");
    PERR("        modelname: resnet/vggnet\n");
    return -1;
  }
  std::string modelname = argv[1];

  batch_sz = atoi(argv[2]);
  int nr_threads = atoi(argv[3]);
  int nr_iterations = atoi(argv[4]);

  if (modelname == "resnet") {
    resnet_main(batch_sz, nr_threads, nr_iterations);
  } else if (modelname == "vggnet") {
    vggnet_main(batch_sz, nr_threads, nr_iterations);
  } else {
    PERR("modelname(%s) not supported", modelname.c_str());
  }
}
