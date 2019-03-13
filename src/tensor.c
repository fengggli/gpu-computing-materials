#include "tensor.h"
#include "common.h"

#include <stdlib.h>
#include <stdarg.h>


dim_t make_dim(int ndims, ...){
  int i;
  va_list vl;
  dim_t dim;
  va_start(vl, ndims);
  for(i = 0; i< MAX_DIM; i++){
    if(i< ndims)
      dim.dims[i] =va_arg(vl, int);
    else
      dim.dims[i] = 0;
  }
  va_end(vl);
  return dim;
}

uint dim_get_capacity(dim_t dim){
  int i;
  uint size = 1;
  for(i = 0; i< MAX_DIM; i++){
    uint tmp = dim.dims[i];
    if(tmp > 0)
      size *= tmp;
    else
      break ;
  }
}

uint dim_get_ndims(dim_t dim){
  int i;
  uint ndims = 0;
  for(i = 0; i< MAX_DIM; i++){
    uint tmp = dim.dims[i];
    if(tmp > 0)
      ndims ++;
    else
      break ;
  }
  return ndims;
}

tensor_t tensor_make(uint const shape[], uint const len){

  int num_elem, i;
  dim_t dim;
  num_elem = 1;

  for(i = 0; i< len; i++){
    assert(shape[i]>0);
    dim.dims[i] = shape[i];
    num_elem*=shape[i];
  }
  tensor_t t;
  t.data =malloc(num_elem*sizeof(T));
  t.dim = dim;
  assert(NULL != t.data);

}



void tensor_destroy(tensor_t t){
  if(t.data){
    free(t.data);
  }
}

void tensor_plus(tensor_t to, tensor_t from){
}

