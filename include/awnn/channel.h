#pragma once

#include "awnn/tensor.h"

/*
 * @breif channel_mean
 *
 * @param t 
 * @param size
 
 */
T channel_mean(T const *t, uint size){
  T mean = 0;
  for (uint i=0; i < size; ++i){
    mean += t[i];
  }
  return mean/size;
}

uint channel_capacity(tensor_t t){
  return t.dim.dims[2] * t.dim.dims[3];
}