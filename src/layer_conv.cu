#include "awnn/layer_conv.h"
#include "cuda_defs.h"

/*
 * For the forward operation, there are a number of "hard" operations for the GPU.
 *
 * The primary one is the transformation of the input array to a 2D array through
 * the im2col process.  This process needs.
 *
 *    * padding
 *    * im2coll
 *
 * Additional critical operations that should be enabled on the GPU are
 *
 *    * tensor transpose 3012
 *    * dot product (gonna use cublas here most likely)
 */




/*
// used for forward
tensor_t tensor_make_transpose_3012(tensor_t t) {
  uint target_idx = 0;
  tensor_t tpose = tensor_make_copy(t);
  for (uint i = 0; i < t.dim.dims[3]; ++i) {  // for each of the new dim 0
    for (uint j = 0; j < t.dim.dims[0] * t.dim.dims[1] * t.dim.dims[2]; ++j) {
      tpose.data[target_idx++] = t.data[i + j * t.dim.dims[3]];
    }
  }
  uint const shape[] = { t.dim.dims[3], t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] };
  tensor_reshape_(&tpose, shape, ARRAY_SIZE(shape));
  return tpose;
}

*/
static __global__ void _do_tensor_make_transpose_3012_device(tensor_t d_transpose, tensor_t d_src) {
  printf("%d\n", threadIdx.x);
}

tensor_t tensor_make_transpose_3012_device(tensor_t t) {
  uint const transposed_shape[] = { t.dim.dims[3], t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] };

  tensor_t d_src = tensor_make_copy_h2d(t);

  tensor_t d_transposed = tensor_make_copy_h2d(t);
  tensor_reshape_(&d_transposed, transposed_shape, ARRAY_SIZE(transposed_shape));

  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");
  _do_tensor_make_transpose_3012_device<<<blocks, threads>>>(d_transposed, d_src);

  tensor_t h_transposed = tensor_make(transposed_shape, ARRAY_SIZE(transposed_shape));
  tensor_copy_d2h(h_transposed, d_transposed);
}




/**
 * TODO
 *
 * This function should just do the padding operation in parallel.  Although below in my notes of the
 * inner im2col, I speculate that this operation can be eliminated, I am still going to provide it in
 * the intitial cuda implementation
 */
static __global__ void _do_tensor_make_padded_square_input_device(tensor_t padded, tensor_t src, uint p, T val)
{
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_padded_square_input_device\n", threadIdx.x);
  }
}

tensor_t tensor_make_padded_square_input_device(tensor_t t, uint p, T val) {

  uint padded_shape[] = { t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] + 2 * p, t.dim.dims[3] + 2 * p };
  tensor_t d_padded = tensor_make_device(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_t d_src = tensor_make_copy_h2d(t);

  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  _do_tensor_make_padded_square_input_device<<<blocks, threads>>>(d_padded, d_src, p, val);

  tensor_t h_padded = tensor_make(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_copy_d2h(h_padded, d_padded);

  return h_padded;
}





static __global__ void _do_forward_device(T *x, uint num_image, uint num_channel,
                                          uint channel_capacity, T *y)
{
  if (threadIdx.x == 0) {
    printf("entered _do_forward_device\n", threadIdx.x);
  }
}
/*
 * primary entry point
 */
status_t convolution_forward_device(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y) {

  return S_ERR;
}



/*
 * This function just sets up the im2col.
 */
tensor_t im2col_device(tensor_t const x, tensor_t const w, conv_param_t const params);




/**
 * TODO
 *
 * The inner should take the x_padded tensor and spread it out over the cols array
 * based on the stride and filter size.  Since we are doing the convolution with the
 * gemm approach, this is a core function.
 *
 * It should generally to how the convolution would work except that it is just doing
 * a copy instead of doing multiplication.  My suspicion is that the bulk of the
 * work from the convolution could be done here, and I'm not currently sure why
 * we would see an improvement in speed from GEMM.  Additionally, note that this
 * function takes in the padded array.  The padded array is used to allow a normal
 * stride in the loops in this fucntino, but it seems like the padded array could be
 * eliminated and this function could manage the padding elements as part of its
 * looping structure.
 *
 * If two loops are created (one for core elements, and one for padding elements)
 * a conditional could be avoided.
 */
__global__ void _do_im2col_inner_device(tensor_t cols, tensor_t x_padded,
    uint N,  uint C,  uint H,  uint W,  uint HH, uint WW,
    uint filter_height, uint filter_width, uint padding, uint stride)
{
  printf("%d\n", threadIdx.x);
}

/**
 * im2col_inner_device is a setup function for the real call to actually launch the kernel.
 * For now, it will allocate and de-allocate / transfer mem to and from the GPU. In the pure
 * GPU based forward, this function will not be called, but rather the _do... function will be
 * called directly.
 */
status_t im2col_inner_device(tensor_t cols, tensor_t x_padded,
                             uint N,  uint C,  uint H,  uint W,  uint HH, uint WW,
                             uint filter_height, uint filter_width, uint padding, uint stride)
{

  tensor_t d_cols       = tensor_make_copy_h2d(cols);
  tensor_t d_x_padded   = tensor_make_copy_h2d(x_padded);

  // TODO: make it handler lager size
  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  _do_im2col_inner_device<<<blocks, threads>>>(d_cols, d_x_padded, N, C, H, W, HH, WW, filter_height, filter_width, padding, stride);

  tensor_copy_d2h(cols, d_cols);

  tensor_destroy_device(&d_cols);
  tensor_destroy_device(&d_x_padded);

  return S_ERR;
}




































status_t convolution_backward_device(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout);


tensor_t col2im_device(tensor_t cols,
                       uint N, uint C, uint H, uint W,
                       uint field_height, uint field_width, uint padding, uint stride);

void col2im_inner_device(tensor_t cols, tensor_t x_padded,
                         uint N, uint C, uint H, uint W, uint HH, uint WW,
                         uint field_height, uint field_width, uint padding, uint stride);