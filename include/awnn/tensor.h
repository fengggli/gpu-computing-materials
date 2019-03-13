/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef TENSOR_H_
#define TENSOR_H_

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
dim_t make_dim(int ndims, ...);
uint dim_get_capacity(dim_t dim);
uint dim_get_ndims(dim_t dim);

typedef struct tensor{
 dim_t dim;
 T *data;
} tensor_t;// tensor

tensor_t tensor_make(uint const shape[], uint const ndims);


// TODO: do nothing
static void _tensor_fill_random(tensor_t t) {}

/* @brief fill a tensor with single scalar*/
static void _tensor_fill_scalar(tensor_t t, T s) {}

tensor_t tensor_make_random(uint const shape[], uint const ndims);

/* @brief Add tensor 'from' to 'to'
 *
 * Notes: this will use the dims of the two input tensor to figure out which calculation to perform
 * e.g.
 *  * x.dims = {3,4,5}, y.dims = [3,4,5], perform an elemwised add to x[:,:,:]
 *  * x.dims = {3,4,5}, y.dims = [4,5], add y to each of the 3 planes with shape[4,5].
 *  * x.dims = {3,4,5}, y.dims = [5], add y to each of with vector with size [5]
 *  * x.dims = {3,4,5}, y.dims = [0], add y to each of the elem
 */
void tensor_plus_(tensor_t x, tensor_t from);
void tensor_dot(tensor_t in1, tensor_t in2, tensor_t out);

/*
 * @brief create tensor of shape, filled with single scalar
 */
tensor_t tensor_make_scalar(uint const shape[], uint const ndims, T s);

#ifdef __cplusplus
}
#endif

#endif
