#include "tensor.h"
#include "common.h"

#include <stdlib.h>


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
