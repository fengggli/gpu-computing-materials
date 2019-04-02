/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include <assert.h>
#include <awnn/logging.h>
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int uint;
// typedef float T;
typedef double T;

typedef unsigned int label_t;
label_t *label_make_random(uint nr_elem, uint range);
void label_destroy(label_t *labels);

enum { MAX_STR_LENGTH = 81 };

typedef int status_t;
enum ERROR_CODE { S_OK = 0, S_ERR = -1, S_BAD_DIM = -2 };

typedef int awnn_mode_t;
enum {
  MODE_TRAIN = 0,
  MODE_INFER = 1,
};

#define AWNN_CHECK_EQ(a, b) \
  if ((a) != (b)) PERR("Expect equal value, but not");

#define AWNN_CHECK_NE(a, b) \
  if ((a) == (b)) PERR("Expect unequal value, but not");

// great than
#define AWNN_CHECK_GT(a, b) \
  if ((a) <= (b)) PERR("Expect lh > rh, but not");

#ifdef __cplusplus
}
#endif
