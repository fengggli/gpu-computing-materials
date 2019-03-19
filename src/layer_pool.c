#include "awnn/layer_pool.h"
#include "awnn/channel.h"

status_t global_avg_pool_forward(tensor_t const x, lcache_t *cache, tensor_t y){
	uint num_images = x.dim.dims[0];
	uint num_channels = x.dim.dims[1];

	for (uint i=0; i < num_images; ++i)
	  for (uint j=0; j < num_channels; ++j){
	  	channel_mean(x.data + i * num_channels * channel_capacity(x) + j * channel_capacity(x)
	  		, channel_capacity);
	  }
}

status_t global_avg_pool_backward(tensor_t dx, lcache_t const *cache, tensor_t const dy){
}
