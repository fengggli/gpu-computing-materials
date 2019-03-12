/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef LAYER_CONV_H_
#define LAYER_CONV_H_

#include "tensor.h"

typedef struct{
  int stride;
  int padding;
} conv_param_t;

/*
 * @param 
 */
void convolution_forward(tensor_t input, tensor_t output, tensor_t filters, conv_param_t params);

void convolution_backward(tensor_t dout, tensor_t* caches, tensor_t dinput, tensor_t dfilters);

#endif
