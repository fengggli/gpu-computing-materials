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
// typedef float T;
typedef double T;
#define T_MIN (-1000.)

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

typedef enum {
  TENSOR_OP_ADD = 0,
  TENSOR_OP_SUB = 1,
  TENSOR_OP_MUL = 2,
  TENSOR_OP_DIV = 3,
} tensor_op_t;

#define tensor_get_capacity(x) (dim_get_capacity((x).dim))
#define tensor_get_ndims(x) (dim_get_ndims((x).dim))
T tensor_get_sum(tensor_t t);

tensor_t tensor_make(uint const shape[], uint const ndims);
void tensor_destroy(tensor_t t);

// TODO: fill random values
static void _tensor_fill_random(tensor_t t, uint seed);
void _tensor_fill_patterned(tensor_t t); // debug use

/* @brief fill tensor buffer with list of values
 *
 * Value_list length needs to be no larger than tensor capacity*/
void tensor_fill_list(tensor_t const, T const value_list[],
                      uint const length_of_value_list);

tensor_t tensor_make_zeros(uint const shape[], uint const ndims);
tensor_t tensor_make_ones(uint const shape[], uint const ndims);
tensor_t tensor_make_random(uint const shape[], uint const ndims, int seed);
tensor_t tensor_make_patterned(uint const shape[], uint const ndims);
tensor_t tensor_make_linspace(T const start, T const stop, uint const shape[], uint const ndims);
/* a new tensor, and it has same shape as the original */
tensor_t tensor_make_alike(tensor_t t);
/* a new tensor, and it has same shape as the original, and it's filled with
 * linspace */
tensor_t tensor_make_linspace_alike(T const start, T const stop,
                                    tensor_t const t);
/* a new tensor, and it's a copy of the original */
tensor_t tensor_make_copy(tensor_t t);
/* a new tensor, and it's a transpose of the original */
tensor_t tensor_make_transpose(tensor_t const t);
/* a new tensor, and it's a sum over one axis */
tensor_t tensor_make_sum(tensor_t const t, uint const axis_id);
/* @brief create tensor of shape, filled with single scalar */
tensor_t tensor_make_scalar(uint const shape[], uint const ndims, T s);

tensor_t tensor_make_padded_square_input(tensor_t t, uint p);
tensor_t tensor_make_scalar_alike(tensor_t t, T scalar);
tensor_t tensor_make_empty_with_dim(dim_t dim);

/* access elem*/
T* tensor_get_elem_ptr(tensor_t const t, dim_t const loc);

void tensor_dump(tensor_t t);

T tensor_rel_error(tensor_t x, tensor_t y);
status_t tensor_reshape_(tensor_t *ptr_t, uint const shape[], uint const ndims);
status_t tensor_reshape_flat_(tensor_t * t);

status_t tensor_elemwise_op_inplace(tensor_t to, tensor_t from, tensor_op_t op);

status_t tensor_add_sameshape(tensor_t in1, tensor_t in2, tensor_t out);
status_t tensor_add_vector_inplace(tensor_t t, tensor_t v);
status_t tensor_matmul(tensor_t in1, tensor_t in2,
                       tensor_t out);                // mm for 2d matrix
status_t tensor_copy(tensor_t to, tensor_t from); // copy, only with same dim

void tensor_print_flat(tensor_t t);

/* some fundamental func*/
static inline void _add(T *to, T *from, uint len) {
  uint i;
  for (i = 0; i < len; i++) {
    to[i] += from[i];
  }
}
static inline void _sub(T *to, T *from, uint len) {
  uint i;
  for (i = 0; i < len; i++) {
    to[i] -= from[i];
  }
}
static inline void _mul(T *to, T *from, uint len) {
  uint i;
  for (i = 0; i < len; i++) {
    to[i] *= from[i];
  }
}
static inline void _div(T *to, T *from, uint len) {
  uint i;
  for (i = 0; i < len; i++) {
    if (from[i] != 0)
      to[i] /= from[i];
  }
}

#ifdef __cplusplus
}
#endif
