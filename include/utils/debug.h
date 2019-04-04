/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include "utils/list.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Obtain a backtrace and print it to stdout. */
static void print_trace (void)
{
  void *array[10];
  size_t size;
  char **strings;
  size_t i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);

  printf ("Obtained %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
     printf ("%s\n", strings[i]);

  free (strings);
}

static inline int list_get_count(struct list_head *head) {
  struct list_head *pos;
  int count = 0;
  list_for_each(pos, head) {
    PSTR("[%d]%p\t", count, pos);
    ++count;
  }
  PSTR("\n");
  return count;
}

#ifdef __cplusplus
}
#endif
#endif
