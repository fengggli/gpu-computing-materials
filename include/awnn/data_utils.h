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
  uint nr_imgs;
  T *data;
  label_t *label;
} dataset_t;

typedef struct {
  uint batch_sz;

  // from orignal data, points to allocated mem
  T * data_train;
  T * data_test;
  label_t * label_train;
  label_t * label_test;

  // Used to split data above to train set/ validation set;
  uint train_split; // [0, train_split) wll be train set
  uint val_split; // [val_split, 50000) will be val set

} data_loader_t;

#ifdef __cplusplus
}
#endif
