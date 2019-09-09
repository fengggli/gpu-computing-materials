1. Example implementation, give interface of convolution.
  1. nnpack 
  2. cudnn
2. Example to use C in CNN
  https://github.com/attractivechaos/kann/blob/master/doc/02dev.md#implementing-convolution

tensor{
 int dim;
 int *dims;
 float *data;
] tensor_t; // tensor

// @brief forward
void convolution_forward(tensor_t input, tensor_t output,tensor filters,conv_param_t)
  // should save intermidate useful information to cache

// @breif backward
convolution_backward(dout, caches, tensor_t dinput, tensor_t dfilters).

after you get your gradients, update your weights, using optmizer like sgd.


