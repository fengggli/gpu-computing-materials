#include "layers/layer_common.hpp"
#include "awnn/logging.h"
#include "pthreadpool.h"

/** Global topology Info*/
struct Topology{
  pthreadpool_t threadpool;

  Topology(int nr_threads ){
    threadpool = pthreadpool_create(nr_threads);
    PINF("launching %d threads", nr_threads);
  }
  ~Topology(){
    pthreadpool_destroy(threadpool);
    PINF("thread pool existing.. ");
    }
};
typedef struct Topology topo_config_t;


/** How each layer is parallized*/
typedef enum {
  PARAL_TYPE_BATCH,
  PARAL_TYPE_MODEL_FC,
  PARAL_TYPE_MODEL_CONV
} paral_config_t;


/** Initialize this layer with machine topology and parallel policy*/
layer_t *layer_setup_hybrid(layer_type_t type, void *layer_config,
                     layer_t *bottom_layer, topo_config_t *topo,  paral_config_t *paral_config){}

/** Destroy this layer*/
void layer_teardown(layer_t *this_layer){}
