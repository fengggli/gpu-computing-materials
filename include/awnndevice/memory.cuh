//
// Created by cmgoebel on 5/6/19.
//

#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"

#include <stdlib.h>

int mem_alloc_device(tensor_t *d_t);

int mem_free_device(tensor_t *d_t);

#ifdef GLOBAL_COUNT_TENSOR_ALLOC_DEALLOC
int INC_TOTAL_TENSOR_ALLOC_DEVICE();
int INC_TOTAL_TENSOR_DEALLOC_DEVICE();

int GET_TOTAL_TENSOR_ALLOC_DEVICE();
int GET_TOTAL_TENSOR_DEALLOC_DEVICE();


void print_memory_alloc_dealloc_totals_device();
int reset_TOTAL_TENSOR_ALLOC_DEVICE();
int reset_TOTAL_TENSOR_DEALLOC_DEVICE();
void reset_all_tensor_device_alloc_dealloc_stats_device();
#endif