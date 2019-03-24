#include "awnn/layer_pool.h"
#include "awnn/channel.h"

// y: N, C, 1, 1
status_t global_avg_pool_forward(tensor_t const x, lcache_t *cache, tensor_t y){
  uint num_images = x.dim.dims[0];
  uint num_channels = x.dim.dims[1];

  for (uint i = 0; i < num_images; ++i)
    for (uint j = 0; j < num_channels; ++j){
      y.data[i * num_channels+j] = channel_mean(x.data + i * num_channels * channel_capacity(x) + j * channel_capacity(x)
        , channel_capacity(x));
  }

  // create cache
  if(cache){
    tensor_t t = tensor_make_empty_with_dim(x.dim);

    cache->all_tensors[cache->count++] = t;
  }   
  
  return S_OK;
}

status_t global_avg_pool_backward(tensor_t dx, lcache_t *cache, tensor_t const dy){
  tensor_t t = cache->all_tensors[--cache->count];
  uint N = t.dim.dims[0];
  uint C = t.dim.dims[1];
  uint H = t.dim.dims[2];
  uint W = t.dim.dims[3];

  float scale_by = 1.0 / (H * W);
//  tensor_t scales = tensor_make_scalar_alike(t, scale_by);

//  tensor_elemwise_op_inplace(scales, dy, TENSOR_OP_MUL);
  for (uint i = 0; i < N ; ++i)
    for(uint j = 0; j < C; ++j)
      for(uint k = 0; k < H; ++k)
        for(uint l = 0; l < W; ++l)
            dx.data[i * C * H * W + j * H * W + k * W + l] *= scale_by * dy.data[i * C + j];

end:
  // free layer cache
  free_lcache(cache);

  return S_OK;
}
