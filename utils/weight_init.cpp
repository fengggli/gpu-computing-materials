//
// Created by lifeng on 4/8/19.
//
#include <math.h>
#include <random>
#include <string>
#include "utils/weight_init.h"

status_t weight_init_linspace(tensor_t t, T const start, T const stop) {
  tensor_fill_linspace(t, start, stop);
  return S_OK;
}

// kaiming init for fc
status_t weight_init_fc_kaiming(tensor_t weight, tensor_t bias) {
  int seed = 1234;
  int fan_in = weight.dim.dims[0];
  // int fan_out = weight.dim.dims[1];

  double gain = sqrt(2.0 / (1 + 0.01 * 0.01));
  double std = gain / sqrt(fan_in);
  double bound =
      sqrt(3.0) * std;  // Calculate uniform bounds from standard deviation
  PINF("-----weight init with uniform (-%f,%f)", bound, bound);

  tensor_fill_random_uniform(weight, -1 * bound, bound, seed);
  PINF("-----bias init with uniform (-%f,%f)", bound, bound);

  bound = 1 / sqrt(fan_in);
  tensor_fill_random_uniform(bias, -1 * bound, bound, seed);
  return S_OK;
}

status_t weight_init_fc(tensor_t weight, tensor_t bias, T weight_scale) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, weight_scale);

  int capacity = tensor_get_capacity(weight);

  int i;
  for (i = 0; i < capacity; i++) {
    double this_value = distribution(generator);  // 0~1
    weight.data[i] = this_value;
  }
  // _tensor_fill_random_normal(weight, 0.0, weight_scale);
  tensor_fill_scalar(bias, 0.0);
  return S_OK;
}

status_t weight_init_kaiming(tensor_t weight) {
  std::default_random_engine generator;

  int N = weight.dim.dims[0];
  // int C = weight.dim.dims[1];
  int H = weight.dim.dims[2];
  int W = weight.dim.dims[3];
  int fan_out = N * H * W;

  T weight_scale = sqrt(2.0 / fan_out);
  std::normal_distribution<double> distribution(0, weight_scale);
  PINF("-----weight init with norm (0, %.3f^2)", weight_scale);

  int capacity = tensor_get_capacity(weight);

  int i;
  for (i = 0; i < capacity; i++) {
    double this_value = distribution(generator);  // 0~1
    weight.data[i] = this_value;
  }

  return S_OK;
}
