/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_COMMON_HPP_
#define LAYER_COMMON_HPP_

#include "awnn/tensor.h"
#include <string>
#include <vector>
#include <stack>

typedef enum{
  LAYER_TYPE_DATA,
  LAYER_TYPE_CONV2D,
  LAYER_TYPE_RELU,
  LAYER_TYPE_FC,
  LAYER_TYPE_SOFTMAX,
  LAYER_TYPE_UNDEFINED
} layer_type_t;

struct Blob{
  // uint id_param;
  std::string name;
  tensor_t data;
  tensor_t diff;

  // TODO: This can be saved in disk actually.
  int learnable;
  tensor_t velocity;  // For momemtum sgd

  Blob(std::string blobname, int learnable, uint shape[4]): learnable(learnable){
    name = blobname;
    data = tensor_make(shape, 4);
    diff = tensor_make_alike(data);
    if(learnable){
      velocity = tensor_make_alike(data);
    }
    else{
      velocity = tensor_make_placeholder(data.dim.dims, tensor_get_ndims(data));
    }
  }

  ~Blob(){
    PINF("now destroy tensor %s, at %p", name.c_str(), data.data);
    tensor_destroy(&data);
    tensor_destroy(&diff);
    if(learnable){
      tensor_destroy(&velocity);
    }
  }
};

struct layer_data_config_t{
  std::string name;
  dim_t dim;
} ;

struct layer_fc_config_t{
  std::string name;
  uint nr_classes;
} ;


struct layer_conv2d_config_t{
  std::string name;

  int stride = 1;
  int padding = 1;
  uint out_channels;
  uint kernel_size;
} ;

typedef struct{
  layer_type_t layer_type = LAYER_TYPE_UNDEFINED;
  std::string name;
  void *config;

  Blob *layer_in;
  Blob *layer_out;

  std::vector<Blob *>learnables;

  std::vector<tensor_t*> tape;
  std::vector<tensor_t> worker_buffer;


  /* all other tensers shall reference in tape (e.g. w, b, or temp)*/
  void (*forward)(tensor_t x,  std::vector<tensor_t*> &tape, tensor_t y, void* layer_config);
  void (*backward)(tensor_t dx,  std::vector<tensor_t*> &tape, tensor_t dy, void * layer_config);
} layer_t;

/** Initialize this layer*/
layer_t* layer_setup(layer_type_t type, void * layer_config, layer_t *bottom_layer);

/** Destroy this layer*/
void layer_teardown(layer_t * this_layer);

typedef struct{
  std::vector<layer_t *> layers;
}net_t;

/** Register this layer to net*/
void net_add_layer(net_t *model, layer_t *layer);

/** Free all layers from net*/
void net_teardown(net_t *net);

/** Forward*/
void net_forward(net_t *net);

/** Backward*/
void net_backward(net_t *net);

void net_update_weights(net_t *net, double learning_rate);

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
#endif
