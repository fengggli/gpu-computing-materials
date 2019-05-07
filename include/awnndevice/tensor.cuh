//
// Created by cmgoebel on 5/7/19.
//

#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"

tensor_t tensor_make_empty_device(dim_t dim);

/* create a tensor directly on the device from a shape */
tensor_t tensor_make_device(int const shape[], int const ndims);

tensor_t tensor_make_alike_device(tensor_t t); // d2d make alike

tensor_t tensor_make_zeros_device(int const shape[], int const ndims);

/* Allocate a tensor in GPU, based on the value from a host tensor*/
tensor_t tensor_make_copy_h2d(tensor_t t_host);

/* destroy the tensor in  GPU*/
void tensor_destroy_device(tensor_t *t);


/* copy the tensor back from gpu to host*/
void tensor_copy_d2h(tensor_t t_host, tensor_t t_device);