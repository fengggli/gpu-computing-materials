//
// Created by lifen on 3/25/19.
//

#include "awnn/net_mlp.h"
#include "awnn/tensor.h"
#include "awnn/layer_fc.h"
#include "awnn/loss_softmax.h"
#include <stdlib.h>
#include <string.h>



status_t mlp_init(model_t *model, uint max_batch_sz,
    uint input_dim, uint output_dim,
    uint nr_hidden_layers, uint hidden_dims[], T reg) {

  status_t ret = S_ERR;

  // save config
  model->max_batch_sz = max_batch_sz;
  model->input_dim = input_dim;
  model->output_dim = output_dim;
  model->nr_hidden_layers = nr_hidden_layers;
  model->reg = reg;
  for(uint i = 0; i< nr_hidden_layers; ++i) model->hidden_dims[i] = hidden_dims[i];

  // init all list structure
  init_list_head(model->list_all_params);
  init_list_head(model->list_layer_out);
  init_list_head(model->list_layer_in);
  init_list_head(model->list_layer_cache);

  // copy of the output tensors of bottom layer
  tensor_t out_prev;
  tensor_t dout_prev;


  for(uint i = 0; i< nr_hidden_layers +1; ++i) {
    uint fan_in, fan_out;
    fan_in = (i == 0 ? input_dim: hidden_dims[i-1]);
    fan_out = (i == nr_hidden_layers ? output_dim: hidden_dims[i]);

    // prepare input
    tensor_t in;
    tensor_t din;
    if(i == 0){
      uint in_shape[] = {max_batch_sz, fan_in};
      in = tensor_make_placeholder();
      din = tensor_make(in_shape, 2);
    }else{
      in = out_prev;
      din = dout_prev;
    }
    char in_name[MAX_STR_LENGTH];
    snprintf(in_name, MAX_STR_LENGTH,"in%u",i );
    net_attach_param(model->list_layer_in, in_name, in, din);

    // prepare weights
    uint w_shape[] = {fan_in, fan_out};
    tensor_t  w= tensor_make(w_shape, 2);
    tensor_t  dw= tensor_make(w_shape, 2);
    char w_name[MAX_STR_LENGTH];
    snprintf(w_name, MAX_STR_LENGTH,"W%u",i );
    net_attach_param(model->list_all_params, w_name, w, dw);

    // prepare bias
    uint b_shape[] = {fan_out};
    tensor_t b= tensor_make(b_shape, 1);
    tensor_t db= tensor_make(b_shape, 1);
    char b_name[MAX_STR_LENGTH];
    snprintf(b_name, MAX_STR_LENGTH,"b%u",i );
    net_attach_param(model->list_all_params, b_name, b, db);

    // prepare layer output
    uint out_shape[] = {max_batch_sz, fan_out};
    tensor_t out = tensor_make(out_shape, 2);
    tensor_t dout = tensor_make(out_shape, 2);
    char out_name[MAX_STR_LENGTH];
    snprintf(out_name, MAX_STR_LENGTH,"out%u",i );
    net_attach_param(model->list_layer_out, out_name, out, dout);

    // prepare layer cache
    // Currently the actual allocation is managed by the fc layer itself
    char cache_name[MAX_STR_LENGTH];
    snprintf(cache_name, MAX_STR_LENGTH,"cache%u",i );
    net_attach_cache(model->list_layer_cache, cache_name);

    // save for higher reference
    out_prev = out;
    dout_prev = dout;
  }
  ret = S_OK;
  return ret;
}


status_t mlp_finalize(model_t *model){
 
  // free all of them
  // net_print_params(model->list_all_params);
    
  net_free_cache(model->list_layer_cache);
  net_free_params(model->list_layer_out); 
  //net_free_params(model->list_layer_in); // TODO: fix the double free here.
  net_free_params(model->list_all_params);
}

tensor_t mlp_scores(model_t const *model, tensor_t x){
  tensor_t layer_input = x;
  tensor_t layer_out;
  for( uint i = 0 ; i< model->nr_hidden_layers+1; i++) {

    char w_name[MAX_STR_LENGTH];
    char b_name[MAX_STR_LENGTH];
    snprintf(w_name, MAX_STR_LENGTH, "W%u", i);
    snprintf(b_name, MAX_STR_LENGTH, "b%u", i);
    tensor_t w = net_get_param(model->list_all_params, w_name)->data;
    tensor_t b = net_get_param(model->list_all_params, b_name)->data;

    // locate preallocated layer_out
    char out_name[MAX_STR_LENGTH];
    snprintf(out_name, MAX_STR_LENGTH,"out%u",i );
    layer_out = net_get_param(model->list_layer_out, out_name)->data;

    // locate preallocated cache
    lcache_t *cache;
    char cache_name[MAX_STR_LENGTH];
    snprintf(cache_name, MAX_STR_LENGTH,"cache%u",i );
    cache = net_get_cache(model->list_layer_cache, cache_name);

    // TODO: need to track y and cache;
    AWNN_CHECK_EQ(S_OK, layer_fc_forward(layer_input, w, b, cache, layer_out));
    layer_input = layer_out;
  }
  return layer_out;
}

/* Compute loss for a batch of (x,y), do forward/backward, and update gradients*/
status_t mlp_loss(model_t const *model, tensor_t x, label_t const *labels, T * ptr_loss){
  *ptr_loss = 0;
  tensor_t mlp_scores(model_t const *model, tensor_t x);
  // forward
  tensor_t out;
  tensor_t dout;
  tensor_t din;

  char out_name[MAX_STR_LENGTH]; // find the param (data/diff) for score
  snprintf(out_name, MAX_STR_LENGTH,"out%u", model->nr_hidden_layers);
  PINF("out score is %s", out_name);
  param_t *param_score = net_get_param(model->list_layer_out, out_name);
  AWNN_CHECK_NE(NULL, labels);
  out = param_score->data;
  dout = param_score->diff;

  awnn_mode_t mode = MODE_TRAIN;
  AWNN_CHECK_EQ( S_OK, loss_softmax(out, labels,
                      ptr_loss, mode, dout));

  // backprop
  for(int i =  model->nr_hidden_layers; i>=0; i--) {
    // locate preallocated layer_in gradient
    char in_name[MAX_STR_LENGTH];
    snprintf(in_name, MAX_STR_LENGTH,"in%u",i );
    din = net_get_param(model->list_layer_in, in_name)->diff;

    char w_name[MAX_STR_LENGTH];
    char b_name[MAX_STR_LENGTH];
    snprintf(w_name, MAX_STR_LENGTH, "W%u", i);
    snprintf(b_name, MAX_STR_LENGTH, "b%u", i);
    tensor_t dw = net_get_param(model->list_all_params, w_name)->diff;
    tensor_t db = net_get_param(model->list_all_params, b_name)->diff;

    // locate preallocated layer_out gradient
    char out_name[MAX_STR_LENGTH];
    snprintf(out_name, MAX_STR_LENGTH,"out%u",i );
    dout = net_get_param(model->list_layer_out, out_name)->diff;

    // locate preallocated cache
    lcache_t *cache;
    char cache_name[MAX_STR_LENGTH];
    snprintf(cache_name, MAX_STR_LENGTH,"cache%u",i );
    cache = net_get_cache(model->list_layer_cache, cache_name);

    // TODO: need to track y and cache;
    AWNN_CHECK_EQ(S_OK, layer_fc_backward(din, dw, db, cache, dout));
  }
}

