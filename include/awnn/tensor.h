/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int uint;
typedef float T;

#define MAX_DIM (4) // N, C, H, W

/*
 * @brief Tensor dimensions
 *
 * examples:
 *  * dims = {0}: a scalar
 *  * dims = {2}: a vector of size [2]
 *  * dims = {3,4,5,6}, a tensor with size [3,4,5,6]
 */
typedef struct{
  uint dims[MAX_DIM];
}dim_t;

/* make dimension, works like block() in cuda*/
dim_t make_dim(int ndims, ...); // TODO: can I just use array as the second param?
uint dim_get_capacity(dim_t dim);
uint dim_get_ndims(dim_t dim);
status_t dim_is_same(dim_t, dim_t);
void dim_dump(dim_t dim);

enum{
  TENSOR_NONE = 0,
};

typedef enum{
  CPU_MEM = 0,
  GPU_MEM = 1,
  BAD_MEM = 2,
}memory_type_t;

typedef struct tensor{
 dim_t dim;
 memory_type_t mem_type;
 T *data;
} tensor_t;// tensor

#define tensor_get_capacity(x) (dim_get_capacity((x).dim))
#define tensor_get_ndims(x) (dim_get_ndims((x).dim))

tensor_t tensor_make(uint const shape[], uint const ndims);
void tensor_destroy(tensor_t t);

// TODO: fill random values
static void _tensor_fill_random(tensor_t t) {}

void _tensor_fill_patterned(tensor_t t); // debug use

tensor_t tensor_make_zeros(uint const shape[], uint const ndims);
tensor_t tensor_make_random(uint const shape[], uint const ndims);
tensor_t tensor_make_patterned(uint const shape[], uint const ndims);
tensor_t tensor_make_linspace(T const start, T const stop, uint const shape[], uint const ndims);
/* a new tensor, and it's a copy of the original */
tensor_t tensor_make_copy(tensor_t t);
/* a new tensor, and it's a transpose of the original */
tensor_t tensor_make_transpose(tensor_t const t);
/* a new tensor, and it's a sum over one axis */
tensor_t tensor_make_sum(tensor_t const t, uint const axis_id);
/* @brief create tensor of shape, filled with single scalar */
tensor_t tensor_make_scalar(uint const shape[], uint const ndims, T s);

/* access elem*/
T* tensor_get_elem_ptr(tensor_t const t, dim_t const loc);

void tensor_dump(tensor_t t);
T tensor_rel_error(tensor_t x, tensor_t y);
status_t tensor_reshape_(tensor_t *ptr_t, uint const shape[], uint const ndims);

status_t tensor_add_inplace(tensor_t to, tensor_t from);
status_t tensor_add_sameshape(tensor_t in1, tensor_t in2, tensor_t out);
status_t tensor_add_vector_inplace(tensor_t t, tensor_t v);
status_t tensor_matmul(tensor_t in1, tensor_t in2,
                       tensor_t out);                // mm for 2d matrix
status_t tensor_copy(tensor_t to, tensor_t from); // copy, only with same dim

/* some fundamental func*/
static inline void _add(T *to, T *from, uint len) {
  uint i;
  for (i = 0; i < len; i++) {
    to[i] += from[i];
  }
}

#ifdef __cplusplus
}
#endif
