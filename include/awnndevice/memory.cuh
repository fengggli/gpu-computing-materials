//
// Created by cmgoebel on 5/6/19.
//

#pragma once

#include "awnn/tensor.h"
#include "awnn/common.h"
#include "awnn/logging.h"

#include <stdlib.h>

void* mem_alloc_device(size_t size);

int mem_free_device(void* data);

/* create a tensor directly on the device from a shape */
tensor_t tensor_make_device(int const shape[], int const ndims);

tensor_t tensor_make_alike_device(tensor_t t); // d2d make alike

tensor_t tensor_make_zeros_device(int const shape[], int const ndims);

/* Allocate a tensor in GPU, based on the value from a host tensor*/
tensor_t tensor_make_copy_h2d(tensor_t t_host);

/* destroy the tensor in  GPU*/
void tensor_destroy_device(tensor_t *ptr_t_device);


/* copy the tensor back from gpu to host*/
void tensor_copy_d2h(tensor_t t_host, tensor_t t_device);


#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
int INC_TOTAL_TENSOR_ALLOC_DEVICE();
int INC_TOTAL_TENSOR_DEALLOC_DEVICE();

int GET_TOTAL_TENSOR_ALLOC_DEVICE();
int GET_TOTAL_TENSOR_DEALLOC_DEVICE();


void print_memory_alloc_dealloc_totals();
int reset_TOTAL_TENSOR_ALLOC_DEVICE();
int reset_TOTAL_TENSOR_DEALLOC_DEVICE();
void reset_all_tensor_device_alloc_dealloc_stats();
#endif