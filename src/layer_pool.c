#include "awnn/channel.h"
#include "awnn/layer_pool.h"

void do_global_pool_forward(tensor_t x, tensor_t y) {
  uint num_images = x.dim.dims[0];
  uint num_channels = x.dim.dims[1];

  for (uint i = 0; i < num_images; ++i)
    for (uint j = 0; j < num_channels; ++j) {
      y.data[i * num_channels + j] =
          channel_mean(x.data + i * num_channels * channel_capacity(x) +
                           j * channel_capacity(x),
                       channel_capacity(x));
    }
}

void do_global_pool_backward(tensor_t dx, tensor_t dy) {
  uint N = dx.dim.dims[0];
  uint C = dx.dim.dims[1];
  uint H = dx.dim.dims[2];
  uint W = dx.dim.dims[3];

  T scale_by = 1.0 / (H * W);
  //  tensor_t scales = tensor_make_scalar_alike(t, scale_by);

  //  tensor_elemwise_op_inplace(scales, dy, TENSOR_OP_MUL);
  for (uint i = 0; i < N; ++i)
    for (uint j = 0; j < C; ++j)
      for (uint k = 0; k < H; ++k)
        for (uint l = 0; l < W; ++l)
          dx.data[i * C * H * W + j * H * W + k * W + l] =
              scale_by * dy.data[i * C + j];
}

void do_max_pool_forward(tensor_t x, tensor_t y, uint kernel_size) {
  uint i, j;
  uint x_linear_index, y_linear_index;
  uint image, channel;

  uint in_images = x.dim.dims[0];
  uint in_channels = x.dim.dims[1];
  uint in_height = x.dim.dims[2];
  uint in_width = x.dim.dims[3];
  uint out_height = in_height / kernel_size;
  uint out_width = in_width / kernel_size;

  for (image = 0; image < in_images; image++) {
    for (channel = 0; channel < in_channels; channel++) {
      for (i = 0; i < in_height; i += kernel_size) {
        for (j = 0; j < in_width; j += kernel_size) {
          T max = -1000;
          uint max_ii = 0, max_jj = 0;
          for (uint ii = 0; ii < kernel_size; ii++) {
            for (uint jj = 0; jj < kernel_size; jj++) {
              x_linear_index =
                  (image * in_channels + channel) * (in_height * in_width) +
                  (i + ii) * in_width + (j + jj);

              if (x.data[x_linear_index] > max) {
                max = x.data[x_linear_index];
                max_ii = ii;
                max_jj = jj;
              }
            }
          }
          x_linear_index =
              (image * in_channels + channel) * (in_height * in_width) +
              (i + max_ii) * in_width + (j + max_jj);
          y_linear_index =
              (image * in_channels + channel) * (out_height * out_width) +
              (i / kernel_size) * out_width + j / kernel_size;
          // PINF("setting %d, %d, %d, %d", image, channel, i/kernel_size,
          // j/kernel_size);
          y.data[y_linear_index] = x.data[x_linear_index];
        }
      }
    }
  }
}

void do_max_pool_backward(tensor_t dx, tensor_t dy, uint kernel_size,
                          tensor_t x, tensor_t y) {
  uint i, j;
  uint x_linear_index, y_linear_index;
  uint image, channel;

  uint in_images = x.dim.dims[0];
  uint in_channels = x.dim.dims[1];
  uint in_height = x.dim.dims[2];
  uint in_width = x.dim.dims[3];

  uint out_height = in_height / kernel_size;
  uint out_width = in_width / kernel_size;

  for (image = 0; image < in_images; image++) {
    for (channel = 0; channel < in_channels; channel++) {
      for (i = 0; i < in_height; i += kernel_size) {
        for (j = 0; j < in_width; j += kernel_size) {
          y_linear_index =
              (image * in_channels + channel) * (out_height * out_width) +
              (i / kernel_size) * out_width + j / kernel_size;
          T max_value = y.data[y_linear_index];
          for (uint ii = 0; ii < kernel_size; ii++) {
            for (uint jj = 0; jj < kernel_size; jj++) {
              x_linear_index =
                  (image * in_channels + channel) * (in_height * in_width) +
                  (i + ii) * in_width + (j + jj);

              dx.data[x_linear_index] = x.data[x_linear_index] < max_value
                                            ? 0
                                            : dy.data[y_linear_index];
            }
          }
        }
      }
    }
  }
}

// y: N, C, 1, 1
status_t global_avg_pool_forward(tensor_t const x, lcache_t *cache,
                                 tensor_t y) {
  do_global_pool_forward(x, y);
  return S_OK;
}

status_t global_avg_pool_backward(tensor_t dx, lcache_t *cache,
                                  tensor_t const dy) {
  do_global_pool_backward(dx, dy);

  return S_OK;
}
