#include "awnndevice/memory.cuh"
#include "awnndevice/tensor.cuh"
#include "awnndevice/layer_pool.cuh"

/* Forward kernel */
static __global__ void _do_forward(T *x, int num_image, int num_channel,
                                   int channel_capacity, T *y) {
  int idx =
      blockIdx.x * blockDim.x + threadIdx.x;  // threadIdx=0, threadIdx=1, ...
  if (idx < num_image * num_channel) {        // totally 6imagesx2channels
    T mean = 0;
    T *channel_start = x + idx * channel_capacity;
    for (int i = 0; i < channel_capacity; i++) {
      mean += channel_start[i];
    }
    mean /= channel_capacity;
    y[idx] = mean;
  }
}

// y: N, C, 1, 1
status_t global_avg_pool_forward_device(tensor_t const x, lcache_t *cache,
                                        tensor_t y) {
  int N = x.dim.dims[0];
  int C = x.dim.dims[1];
  int H = x.dim.dims[2];
  int W = x.dim.dims[3];

  tensor_t d_x = tensor_make_copy_h2d(x);
  tensor_t d_y = tensor_make_copy_h2d(y);

  // TODO: make it handler lager size
  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  _do_forward<<<blocks, threads>>>(d_x.data, N, C, H * W, d_y.data);

  if (cache) {
    tensor_t t = tensor_make_empty_with_dim(x.dim);
    lcache_push(cache, t);
  }

  tensor_copy_d2h(y, d_y);

  tensor_destroy_device(&d_x);
  tensor_destroy_device(&d_y);

  return S_OK;
}

/* Backward kernel */
static __global__ void _do_backward(T *dx, int num_image, int num_channel,
                                    int channel_capacity, T *dy) {
  int idx =
      blockIdx.x * blockDim.x + threadIdx.x;  // threadIdx=0, threadIdx=1, ...
  if (idx < num_image * num_channel) {        // totally 6imagesx2channels
    T scale_by = 1.0 / (channel_capacity);
    T *channel_start = dx + idx * channel_capacity;
    for (int i = 0; i < channel_capacity; i++) {
      channel_start[i] = scale_by * dy[idx];
    }
  }
}

status_t global_avg_pool_backward_device(tensor_t dx, lcache_t *cache,
                                         tensor_t const dy) {
  tensor_t t = lcache_pop(cache);
  int N = t.dim.dims[0];
  int C = t.dim.dims[1];
  int H = t.dim.dims[2];
  int W = t.dim.dims[3];

  tensor_t d_dx = tensor_make_copy_h2d(dx);
  tensor_t d_dy = tensor_make_copy_h2d(dy);

  //  tensor_t scales = tensor_make_scalar_alike(t, scale_by);

  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  //  tensor_elemwise_op_inplace(scales, dy, TENSOR_OP_MUL);
  _do_backward<<<blocks, threads>>>(d_dx.data, N, C, H * W, d_dy.data);

  tensor_copy_d2h(dx, d_dx);
  tensor_destroy_device(&d_dx);
  tensor_destroy_device(&d_dy);

  // free layer cache
  tensor_destroy(&t);

  return S_OK;
}
