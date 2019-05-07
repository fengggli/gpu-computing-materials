#pragma once

#include "awnn/tensor.h"

/*
 * @breif channel_mean
 *
 * @param t 
 * @param size
 
 */
T channel_mean(T const *t, int size){
  T mean = 0;
  for (int i=0; i < size; ++i){
    mean += t[i];
  }
  return mean/(T)size;
}

int channel_capacity(tensor_t t){
  return t.dim.dims[2] * t.dim.dims[3];
}