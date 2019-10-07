/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_COMMON_HPP_
#define LAYER_COMMON_HPP_

#include <map>
#include <stack>
#include <string>
#include <vector>
#include "awnn/tensor.h"
#include "utils/data_cifar.h"

typedef enum {
  LAYER_TYPE_DATA,
  LAYER_TYPE_CONV2D,
  LAYER_TYPE_RELU_INPLACE,
  LAYER_TYPE_FC,
  LAYER_TYPE_SOFTMAX,
  LAYER_TYPE_RESBLOCK,
  LAYER_TYPE_POOL,
  LAYER_TYPE_UNDEFINED,
} layer_type_t;

typedef enum {
  ACTIVATION_NONE = -1,
  ACTIVATION_RELU,
} activation_t;

struct Blob {
  // uint id_param;
  std::string name;
  tensor_t data;
  tensor_t diff;

  // TODO: This can be saved in disk actually.
  int learnable;
  tensor_t velocity;  // For momemtum sgd

  Blob(std::string blobname, int learnable, uint shape[4])
      : learnable(learnable) {
    name = blobname;
    data = tensor_make(shape, 4);
    diff = tensor_make_alike(data);
    if (learnable) {
      velocity = tensor_make_alike(data);
    } else {
      velocity = tensor_make_placeholder(data.dim.dims, tensor_get_ndims(data));
    }
  }

  ~Blob() {
    PDBG("now destroy tensor %s, at %p", name.c_str(), data.data);
    tensor_destroy(&data);
    tensor_destroy(&diff);
    if (learnable) {
      tensor_destroy(&velocity);
    }
  }
};

struct layer_data_config_t {
  std::string name;
  dim_t dim;
};

struct layer_relu_config_t {
  std::string name;
};

struct layer_pool_config_t {
  std::string name;
};

struct layer_fc_config_t {
  std::string name;
  uint nr_classes;
  activation_t activation = ACTIVATION_NONE;

  double reg = 0;  // l2 regulizer
};

struct layer_conv2d_config_t {
  std::string name;

  int stride = 1;
  int padding = 1;
  uint out_channels;
  uint kernel_size;
  activation_t activation = ACTIVATION_NONE;

  double reg = 0;  // l2 regulizer
};

struct layer_resblock_config_t {
  std::string name;
  int is_new_stage = 0; /** first resblock in new stage will use stride 2 and
                           double channels in first conv layer*/
  uint kernel_size = 3;

  activation_t activation = ACTIVATION_RELU;

  double reg = 0;  // l2 regulizer
};

using tape_t = std::map<std::string, Blob *>;

typedef struct {
  layer_type_t layer_type = LAYER_TYPE_UNDEFINED;
  std::string name;
  void *config;

  Blob *layer_in;
  Blob *layer_out;

  std::vector<Blob *> learnables;

  tape_t tape;
  std::vector<Blob *> temp_blobs; /*Used for to store ouputs of a sublayer*/

  /* all other tensers shall reference in tape (e.g. w, b, or temp)*/
  double (*forward)(tensor_t x, tape_t &tape, tensor_t y, void *layer_config);
  void (*backward)(tensor_t dx, tape_t &tape, tensor_t dy, void *layer_config);
} layer_t;

/** Initialize this layer*/
layer_t *layer_setup(layer_type_t type, void *layer_config,
                     layer_t *bottom_layer);

/** Destroy this layer*/
void layer_teardown(layer_t *this_layer);

typedef struct {
  std::vector<layer_t *> layers;   // data layer is first layer
  layer_data_config_t dataconfig;  // each replica might have different nr_imgs
} net_t;

/** Register this layer to net*/
void net_add_layer(net_t *model, layer_t *layer);

/** Free all layers from net*/
void net_teardown(net_t *net);

/** Forward
 * Return all loss from regulizer*/
double net_forward(net_t *net);

/** Backward*/
void net_backward(net_t *net);

void net_update_weights(net_t *net, double learning_rate);

void net_loss(net_t *net, tensor_t x, label_t const *labels, T *ptr_loss,
              int verbose = 0);

/* Resnet related*/
void resnet_setup(net_t *net, uint input_shape[], double reg);

void resnet_teardown(net_t *net);

/**
 * Multi-thread support.
 * Each read will do model_init
 **/
struct resnet_thread_info {
  int id;
  int nr_threads;
  int nr_iterations;
  net_t model;

  data_loader_t *data_loader;
  uint batch_sz;

  pthread_mutex_t *ptr_mutex;
  pthread_barrier_t *ptr_barrier;
};

typedef struct resnet_thread_info resnet_thread_info_t;

void *resnet_thread_entry(void *threadinfo);

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
#endif