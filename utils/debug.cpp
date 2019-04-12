/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include <cmath>
#include "awnn/logging.h"
#include "utils/debug.h"

void print_trace(void) {
  void *array[10];
  int size;
  char **strings;
  int i;

  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);

  printf ("Obtained %d stack frames.\n", size);

  for (i = 0; i < size; i++) printf("%s\n", strings[i]);

  free(strings);
}

int list_get_count(struct list_head *head) {
  struct list_head *pos;
  int count = 0;
  list_for_each(pos, head) {
    PSTR("[%d]%p\t", count, pos);
    ++count;
  }
  PSTR("\n");
  return count;
}

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
  PNOTICE("[Tensor(%p) %s]: mean = %.2e, std = %.2e", t.data, name, mean, std);
}
