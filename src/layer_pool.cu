#include "awnn/layer_pool.h"

/* this is a */
static __global__ void _do_forward(T *input, uint num_image, uint num_channels,
                                   uint channel_capacity, T *output) {
  // do nothing
  /*  for (uint i = 0; i < num_images; ++i)*/
  /*for (uint j = 0; j < num_channels; ++j) {*/
  /*double mean = 0;*/
  /*for()*/
  /*y.data[i * num_channels + j] =*/
  /*scan(x.data + i * num_channels * channel_capacity +*/
  /*j * channel_capacity,*/
  /*channel_capacity(x));*/
  /*}*/
}

// y: N, C, 1, 1
status_t global_avg_pool_forward_device(tensor_t const x, lcache_t *cache,
                                        tensor_t y) {
  uint num_images = x.dim.dims[0];
  uint num_channels = x.dim.dims[1];
  uint capacity_x = tensor_get_capacity(x);
  uint capacity_y = tensor_get_capacity(y);

  tensor_t d_x = tensor_make_copy_h2d(x);
  tensor_t d_y = tensor_make_copy_h2d(y);

  dim3 threads(32);
  dim3 blocks(1);
  // dim3 blocks(N/threads.x);

  _do_forward<<<blocks, threads>>>(d_x.data, num_images, num_channels,
                                   (capacity_x) / (num_images * num_channels),
                                   d_y.data);

  if (cache) {
    tensor_t t = tensor_make_empty_with_dim(x.dim);
    lcache_push(cache, t);
  }

  tensor_destroy_device(d_x);
  tensor_destroy_device(d_y);

  return S_OK;
}

status_t global_avg_pool_backward_device(tensor_t dx, lcache_t *cache,
                                         tensor_t const dy) {
  tensor_t t = lcache_pop(cache);
  uint N = t.dim.dims[0];
  uint C = t.dim.dims[1];
  uint H = t.dim.dims[2];
  uint W = t.dim.dims[3];

  float scale_by = 1.0 / (H * W);
  //  tensor_t scales = tensor_make_scalar_alike(t, scale_by);

  //  tensor_elemwise_op_inplace(scales, dy, TENSOR_OP_MUL);
  for (uint i = 0; i < N; ++i)
    for (uint j = 0; j < C; ++j)
      for (uint k = 0; k < H; ++k)
        for (uint l = 0; l < W; ++l)
          dx.data[i * C * H * W + j * H * W + k * W + l] =
              scale_by * dy.data[i * C + j];

  // free layer cache
  tensor_destroy(t);

  return S_OK;
}
