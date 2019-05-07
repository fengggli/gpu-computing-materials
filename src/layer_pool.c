#include "awnn/channel.h"
#include "awnn/layer_pool.h"

// y: N, C, 1, 1
status_t global_avg_pool_forward(tensor_t const x, lcache_t *cache,
                                 tensor_t y) {
  int num_images = x.dim.dims[0];
  int num_channels = x.dim.dims[1];

  for (int i = 0; i < num_images; ++i)
    for (int j = 0; j < num_channels; ++j) {
      y.data[i * num_channels + j] =
          channel_mean(x.data + i * num_channels * channel_capacity(x) +
                           j * channel_capacity(x),
                       channel_capacity(x));
    }

  // create cache
  if (cache) {
    tensor_t t = tensor_make_empty_with_dim(x.dim);
    lcache_push(cache, t);
  }

  return S_OK;
}

status_t global_avg_pool_backward(tensor_t dx, lcache_t *cache,
                                  tensor_t const dy) {
  tensor_t t = lcache_pop(cache);
  int N = t.dim.dims[0];
  int C = t.dim.dims[1];
  int H = t.dim.dims[2];
  int W = t.dim.dims[3];

  T scale_by = 1.0 / (H * W);
  //  tensor_t scales = tensor_make_scalar_alike(t, scale_by);

  //  tensor_elemwise_op_inplace(scales, dy, TENSOR_OP_MUL);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < C; ++j)
      for (int k = 0; k < H; ++k)
        for (int l = 0; l < W; ++l)
          dx.data[i * C * H * W + j * H * W + k * W + l] =
              scale_by * dy.data[i * C + j];

  // free layer cache
  tensor_destroy(&t);

  return S_OK;
}
