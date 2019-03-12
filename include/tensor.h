/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef TENSOR_H_
#define TENSOR_H_

typedef unsigned int uint;
typedef float T;

#define MAX_DIM (4) // N, C, H, W

typedef struct{
  uint dims[MAX_DIM]; //{2,2}, {3,4,5,6}
}dim_t;


typedef struct tensor{
 dim_t dim;
 T *data;
} tensor_t;// tensor

tensor_t tensor_make(uint const shape[], uint const len);


// TODO: do noting
static void _tensor_fill_random(tensor_t t);

tensor_t tensor_make_random(uint const shape[], uint const len){
  tensor_t t =  tensor_make(shape, len);
  _tensor_fill_random(t);
  return t;
}

#endif
