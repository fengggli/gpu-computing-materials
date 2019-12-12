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
#include "awnn/layer_common.hpp"
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
  uint batch_sz = 128;

  if(argc != 4){
    PERR("format: cmd batch_size nr_worker_threads nr_iterations");
    return -1;
  }

  batch_sz = atoi(argv[1]);
  int nr_worker_threads = atoi(argv[2]);
  int nr_iterations = atoi(argv[3]);

  /* Data loader*/
  data_loader_t loader;
  status_t ret =
      cifar_open_batched(&loader, CIFAR_PATH, batch_sz, nr_worker_threads);
  uint val_sz = 1000;
  EXPECT_EQ(S_OK, ret);
  EXPECT_EQ(S_OK, cifar_split_train(&loader, train_sz, val_sz));

  pthread_t *threads  = (pthread_t*)malloc(sizeof(pthread_t)*nr_worker_threads);
  resnet_thread_info_t *threads_info =
      new resnet_thread_info_t[nr_worker_threads];
  AWNN_CHECK_NE(threads_info, 0);

  status_t rc = -1;

  /* Initialize and set thread detached attribute */
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_mutex_t mutex;
  pthread_mutex_init(&mutex, NULL);

  /* Used for all-reduce*/
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, nr_worker_threads);

  for(int t = 0;  t< nr_worker_threads; t++){
    threads_info[t].id = t;
    threads_info[t].nr_threads = nr_worker_threads;
    threads_info[t].data_loader = &loader;
    threads_info[t].batch_sz = batch_sz;
    threads_info[t].ptr_mutex = &mutex;
    threads_info[t].ptr_barrier = &barrier;
    threads_info[t].nr_iterations = nr_iterations;

          rc = pthread_create(&threads[t], &attr, resnet_thread_entry, (void *)(&threads_info[t]));
    AWNN_CHECK_EQ(0, rc);
  }
  
  /*
  time_point_t start, end;
  std::vector<double> eclapsed_times;
  uint nr_iterations = 50;
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

  */

	for(int t=0; t<nr_worker_threads; t++) {
    rc = pthread_join(threads[t], NULL);
    AWNN_CHECK_EQ(0, rc);
	}

  PWRN("joined!");
  pthread_barrier_destroy(&barrier);

  free(threads);
  delete[] threads_info;

  pthread_mutex_destroy(&mutex);
  cifar_close(&loader);
}

