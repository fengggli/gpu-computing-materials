/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <assert.h>

typedef int status_t;
enum ERROR_CODE { S_OK = 0, S_ERR = -1, S_BAD_DIM = -2 };

#define ARRAY_SIZE(x)  (sizeof(x) / sizeof((x)[0]))

#endif
