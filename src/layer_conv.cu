#include "awnn/layer_conv.h"

__global__ void _do_forward_device(T *x, uint num_image, uint num_channel,
                                   uint channel_capacity, T *y)
{

}

status_t convolution_forward_device(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y) {

  return S_ERR;
}


tensor_t im2col_device(tensor_t const x, tensor_t const w, conv_param_t const params);



__global__ void _do_im2col_inner_device(tensor_t cols, tensor_t x_padded,
    uint N,  uint C,  uint H,  uint W,  uint HH, uint WW,
    uint filter_height, uint filter_width, uint padding, uint stride)
{


}



/**
 * im2col_inner_device is a setup function for the real call to actually launch the kernel.
 * For now, it will allocate and de-allocate / transfer mem to and from the GPU. In the pure
 * GPU based forward, this function will not be called, but rather the _do... function will be
 * called directly.
 *
 * @param cols
 * @param x_padded
 * @param N
 * @param C
 * @param H
 * @param W
 * @param HH
 * @param WW
 * @param filter_height
 * @param filter_width
 * @param padding
 * @param stride
 * @return
 */
status_t im2col_inner_device(tensor_t cols, tensor_t x_padded,
                             uint N,  uint C,  uint H,  uint W,  uint HH, uint WW,
                             uint filter_height, uint filter_width, uint padding, uint stride)
{


  return S_ERR;
}








status_t convolution_backward_device(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout);


tensor_t col2im_device(tensor_t cols,
                       uint N, uint C, uint H, uint W,
                       uint field_height, uint field_width, uint padding, uint stride);

void col2im_inner_device(tensor_t cols, tensor_t x_padded,
                         uint N, uint C, uint H, uint W, uint HH, uint WW,
                         uint field_height, uint field_width, uint padding, uint stride);