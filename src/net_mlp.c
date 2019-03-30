//
// Created by lifen on 3/25/19.
//

#include "awnn/net_mlp.h"
#include "awnn/tensor.h"
#include "awnn/layer_fc.h"
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
  init_list_head(model->list_layer_cache);


  for(uint i = 0; i< nr_hidden_layers +1; ++i) {
    uint fan_in, fan_out;
    fan_in = (i == 0 ? input_dim: hidden_dims[i-1]);
    fan_out = (i == nr_hidden_layers ? output_dim: hidden_dims[i]);

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
    tensor_t out = tensor_make(out_shape, 1);
    tensor_t dout = tensor_make(out_shape, 1);
    char out_name[MAX_STR_LENGTH];
    snprintf(out_name, MAX_STR_LENGTH,"out%u",i );
    net_attach_param(model->list_layer_out, out_name, out, dout);

    // prepare layer cache
    // Currently the actual allocation is managed by the fc layer itself
    char cache_name[MAX_STR_LENGTH];
    snprintf(cache_name, MAX_STR_LENGTH,"cache%u",i );
    net_attach_cache(model->list_layer_cache, cache_name);
  }
  ret = S_OK;
  return ret;
}


status_t mlp_finalize(model_t *model){
 
  // free all of them
  // net_print_params(model->list_all_params);
    
  net_free_cache(model->list_layer_cache);
  net_free_params(model->list_layer_out);
  net_free_params(model->list_all_params);
}



tensor_t mlp_scores(model_t const *model, tensor_t x){
  for( uint i = 0 ; i< model->nr_hidden_layers+1; i++) {

    char w_name[MAX_STR_LENGTH];
    char b_name[MAX_STR_LENGTH];
    snprintf(w_name, MAX_STR_LENGTH, "W%u", i);
    snprintf(b_name, MAX_STR_LENGTH, "b%u", i);
    tensor_t w = net_get_param(model->list_all_params, w_name)->data;
    tensor_t b = net_get_param(model->list_all_params, w_name)->data;

    uint shape_out[] = {x.dim.dims[0], w.dim.dims[1]};
    tensor_t out = tensor_make(shape_out, 2);
    lcache_t cache;

    // TODO: need to track y and cache;
    AWNN_CHECK_EQ(S_OK, layer_fc_forward(x, w, b, &cache, out));
  }
}
