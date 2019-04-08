#include <math.h>
#include "awnn/weight_init.h"

extern status_t _tensor_fill_linspace(tensor_t t, float const start,
                                      float const stop);
extern void _tensor_fill_random_uniform(tensor_t t, double low, double high,
                                        uint seed);

status_t weight_init_linspace(tensor_t t, T start, T stop) {
  return _tensor_fill_linspace(t, start, stop);
}

// kaiming init for fc
status_t weight_init_fc(tensor_t weight, tensor_t bias) {
  uint seed = 1234;
  uint fan_in = weight.dim.dims[0];
  // uint fan_out = weight.dim.dims[1];

  double gain = sqrt(2.0 / (1 + 0.01 * 0.01));
  double std = gain / sqrt(fan_in);
  double bound =
      sqrt(3.0) * std;  // Calculate uniform bounds from standard deviation
  PINF("weight init with uniform (-%f,%f)", bound, bound);

  _tensor_fill_random_uniform(weight, -1 * bound, bound, seed);
  PINF("bias init with uniform (-%f,%f)", bound, bound);

  bound = 1 / sqrt(fan_in);
  _tensor_fill_random_uniform(bias, -1 * bound, bound, seed);
  return S_OK;
}
