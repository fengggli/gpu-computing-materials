//
// Created by lifen on 3/25/19.
//

#include "awnn/net_mlp.h"

status_t mlp_init(model_t *model, uint max_batch_sz,
    uint input_dim, uint output_dim,
    uint nr_hidden_layers, uint hidden_dims[], T reg) {
  // save config
  model->max_batch_sz = max_batch_sz;
  model->input_dim = input_dim;
  model->output_dim = output_dim;
  model->nr_hidden_layers = nr_hidden_layers;
  model->reg = reg;
  for(uint i = 0; i< nr_hidden_layers; ++i) model->hidden_dims[i] = hidden_dims[i];

  for(uint i = 0; i< nr_hidden_layers +1; ++i) {
    uint fan_in, fan_out;
    fan_in = (i == 0 ? input_dim: hidden_dims[i-1]);
    fan_out = (i == nr_hidden_layers ? output_dim: hidden_dims[i]);

    // allocate this space
    uint shape_w[] = {fan_in, fan_out};
    uint shape_b[] = {fan_out};
    tensor_t  w= tensor_make(shape_w, 2);
    tensor_t  dw= tensor_make(shape_w, 2);

    tensor_t b= tensor_make(shape_b, 1);
    tensor_t db= tensor_make(shape_b, 1);
  }

}

status_t mlp_finalize(model_t *model){
 
  // free all of them

}
