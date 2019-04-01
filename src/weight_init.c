#include "awnn/weight_init.h"

extern status_t _tensor_fill_linspace(tensor_t t, float const start,
                                      float const stop);

status_t weight_init_linspace(tensor_t t, T start, T stop) {
  return _tensor_fill_linspace(t, start, stop);
}
