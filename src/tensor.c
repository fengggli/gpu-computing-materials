#include "awnn/tensor.h"
#include "awnn/logging.h"

#include <stdlib.h>
#include <stdarg.h>

#define SIZE_LINE_BUFFER (160)


dim_t make_dim(int ndims, ...) {
  int i;
  va_list vl;
  dim_t dim;
  va_start(vl, ndims);
  for(i = 0; i< MAX_DIM; i++){
    if(i < ndims)
      dim.dims[i] = va_arg(vl, int);
    else
      dim.dims[i] = 0;
  }
  va_end(vl);
  return dim;
}

uint dim_get_capacity(dim_t dim){
  int i;
  uint size = 1;
  for(i = 0; i < MAX_DIM; i++){
    uint tmp = dim.dims[i];
    if(tmp > 0)
      size *= tmp;
    else
      break ;
  }
  return size;
}

uint dim_get_ndims(dim_t dim){
  int i;
  uint ndims = 0;
  for(i = 0; i < MAX_DIM; i++){
    uint tmp = dim.dims[i];
    if(tmp > 0)
      ndims ++;
    else
      break ;
  }
  return ndims;
}

status_t dim_is_same(dim_t dim1, dim_t dim2){
  uint i;
  for(i = 0; i < MAX_DIM; i++){
    if(dim1.dims[i] != dim2.dims[i]){
      return S_BAD_DIM;
    }
  }
  return S_OK;
}

void dim_dump(dim_t dim){
  int i;
  PSTR("Dimension Dump: [");
  for(i = 0; i< MAX_DIM; i++){
    uint tmp = dim.dims[i];
    if(tmp > 0)
      PSTR("%d ", tmp);
    else
      break ;
  }
  PSTR("]\n");
}


void _tensor_fill_patterned(tensor_t t){
  uint capacity = dim_get_capacity(t.dim);
  uint i;
  for(i = 0; i< capacity; i++){
    t.data[i] = (T)(i);
  }
}

tensor_t _tensor_make(dim_t dim){
  tensor_t t;
  uint capacity;
  capacity = dim_get_capacity(dim);
  t.data =malloc(capacity*sizeof(T));
  t.dim = dim;
  assert(NULL != t.data);
  return t;
}

tensor_t tensor_make(uint const shape[], uint const ndims){
  int i;
  dim_t dim;

  if(ndims == 0){
    PINF("make zero");
    dim = make_dim(0,0);
  }

  for(i = 0; i< MAX_DIM; i++){
    if(i< ndims)
      dim.dims[i] =shape[i];
    else
      dim.dims[i] = 0;
  }
  return _tensor_make(dim);
}

tensor_t tensor_make_random(uint const shape[], uint const ndims){
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_random(t);
  return t;
}

tensor_t tensor_make_copy(tensor_t t){
  tensor_t ret = _tensor_make(t.dim);
  tensor_copy(ret, t);
  return ret;
}


tensor_t tensor_make_scalar(uint const shape[], uint const ndims, T s){
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_scalar(t, s);
  return t;
}

status_t _tensor_fill_linspace(tensor_t t, float const start, float const stop){
  uint i;
  uint capacity = dim_get_capacity(t.dim);
  if(stop <= start) {
    PERR("Wrong linspace");
    return S_ERR;
  }
  T step = (stop - start)/((T)capacity -1);
  for(i = 0; i< capacity; i++) {
    t.data[i] = start + i*step;
  }
}

tensor_t tensor_make_linspace(T const start, T const stop, uint const shape[], uint const ndims){
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_linspace(t, start, stop);
  return t;
}

tensor_t tensor_make_patterned(uint const shape[], uint const ndims){
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_patterned(t);
  return t;
}


void  _dump(T* data, dim_t dim, int cur_dim_id, int cur_capacity){
  uint i;
  for (i =0; i< dim.dims[cur_dim_id]; i++){
    if(cur_dim_id + 1 == dim_get_ndims(dim)){ // this is the vector
      PSTR("%.3f ", data[i]);
    }
    else{
      PSTR("{");
      _dump(data + i*(cur_capacity), dim, cur_dim_id+1, cur_capacity/dim.dims[cur_dim_id+1]);
      PSTR("}\n");
    }

  }
}

void tensor_dump(tensor_t t){
  PINF("Dump tensor\n");
  dim_t dim = t.dim;
  dim_dump(t.dim);
  uint capacity = dim_get_capacity(dim);
  PSTR("{");
  if(dim.dims[0]==0)
    PSTR("%.3f ", t.data[0]); //scalar
  else{
    _dump(t.data, dim, 0, capacity/dim.dims[0]);
  }
  PSTR("}\n");
}

void tensor_destroy(tensor_t t){
  if(t.data){
    free(t.data);
  }
}
