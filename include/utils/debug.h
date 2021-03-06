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
#include "awnn/tensor.h"

typedef struct timespec clocktime_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Obtain a backtrace and print it to stdout. */
void print_trace();
int list_get_count(struct list_head *head);

/** Set handler to print backtrace during segfault*/
void init_helper_env();

/* Show tensor std, variance, etc */
void dump_tensor_stats(tensor_t, const char *name);

clocktime_t get_clocktime();
double get_elapsed_ms(clocktime_t start, clocktime_t end);
#ifdef __cplusplus
}
#endif
#endif
