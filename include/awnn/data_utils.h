//
// Created by lifen on 3/25/19.
//
#pragma once

#include "awnn/common.h"
#include "awnn/data_utils.h"
#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  T data;
} data_t;

// Open data from files
data_t model_open_data();

#ifdef __cplusplus
}
#endif
