/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "config.h"

#include <assert.h>
#include <awnn/logging.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int uint;

#ifdef AWNN_USE_FLT32
typedef float T;
#else
typedef double T;
#endif

typedef unsigned int label_t;
label_t *label_make_random(uint nr_elem, uint range);
void label_destroy(label_t *labels);

enum { MAX_STR_LENGTH = 255 };

typedef int status_t;
enum ERROR_CODE { S_OK = 0, S_ERR = -1, S_BAD_DIM = -2 };

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

typedef int awnn_mode_t;
enum {
  MODE_TRAIN = 0,
  MODE_INFER = 1,
};

void print_trace();

#define AWNN_NO_USE(a) (void)(a)

#define AWNN_CHECK_EQ(a, b)                                         \
  if ((a) != (b)) {                                                           \
    PERR("[%s:%d]: Value (%lu) != %lu", __FILE__, __LINE__, (unsigned long)a, \
         (unsigned long)b);                                                   \
    print_trace();                                                  \
    exit(-1);                                                       \
  }

#define AWNN_CHECK_NE(a, b)                \
  if ((a) == (b)) {                        \
    PERR("Expect unequal value, but not"); \
    exit(-1);                              \
  }

// great than
#define AWNN_CHECK_GT(a, b)          \
  if ((a) <= (b)) {                  \
    PERR("Expect lh > rh, but not"); \
    exit(-1);                        \
  }

#ifdef __cplusplus
}
#endif
