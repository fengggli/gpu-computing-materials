#ifndef LAYER_PARAL_HPP_
#define LAYER_PARAL_HPP_

#include "awnn/logging.h"
#include "pthreadpool.h"

/** Global topology Info*/
struct Topology {
  pthreadpool_t threadpool;
  int nr_threads;

  Topology(int nr_threads) : nr_threads(nr_threads) {
    threadpool = pthreadpool_create(nr_threads);
    PINF("launching %d threads", nr_threads);
  }
  ~Topology() {
    pthreadpool_destroy(threadpool);
    PINF("thread pool existing.. ");
  }
};
typedef struct Topology topo_config_t;

/** How each layer is parallized*/
typedef enum {
  PARAL_TYPE_DATA,
  PARAL_TYPE_MODEL_FC,
  PARAL_TYPE_MODEL_CONV  // not supported yet
} paral_config_t;

#endif
