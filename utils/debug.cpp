/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include <cmath>
#include "awnn/logging.h"
#include "utils/debug.h"

void dump_tensor_stats(tensor_t t, const char* name) {
  uint capacity = tensor_get_capacity(t);
  double sum = 0;
  for (uint i = 0; i < capacity; i++) {
    sum += t.data[i];
  }
  double mean = sum / capacity;

  sum = 0;
  for (uint i = 0; i < capacity; i++) {
    sum += (t.data[i] - mean) * (t.data[i] - mean);
  }
  double std = sqrt(sum / (capacity - 1));
  PNOTICE("[Tensor %s]: mean = %.2e, std = %.2e", name, mean, std);
}
