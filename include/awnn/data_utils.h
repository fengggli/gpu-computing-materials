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
  uint batch_sz;
  T *data_train;
  T *data_test;
  label_t *label_train;
  label_t *label_test;
} data_loader_t;

#ifdef __cplusplus
}
#endif
