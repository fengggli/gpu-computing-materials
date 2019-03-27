#include "awnn/tensor.h"
#include "awnn/logging.h"

#include <stdlib.h>
#include <stdarg.h>

#define SIZE_LINE_BUFFER (160)

// make_dim(3, {2,3,4}):
// TODO: make it more robust
/*dim_t make_dim(uint ndims, uint all_dims[]){*/
/*}*/

dim_t make_dim(int ndims, ...) {
  int i;
  va_list vl;
  dim_t dim;
  va_start(vl, ndims);
  assert(ndims <= MAX_DIM);
  for(i = 0; i< MAX_DIM; i++){
    if(i < ndims)
      dim.dims[i] = va_arg(vl, int);
    else
      dim.dims[i] = 0;
  }
  va_end(vl);
  return dim;
}

dim_t dim_get_reverse(dim_t dim) {
  dim_t ret_dim;
  uint i = 0;
  uint ndims = dim_get_ndims(dim);
  for(i = 0; i< ndims; i++) {
    ret_dim.dims[ndims -i -1] = dim.dims[i];
  }
  for(i = ndims; i< MAX_DIM; i++)
    ret_dim.dims[i] = 0;
  return ret_dim;
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

/*
 * Tensor
 */

T tensor_get_sum(tensor_t t){
  T ret = 0;
  for (uint i = 0; i < tensor_get_capacity(t); i++) {
    ret += t.data[i];
  }
  return ret;
}

void _tensor_fill_random(tensor_t t, uint seed){
  srand(seed);
  uint capacity = dim_get_capacity(t.dim);
  uint i;
  for(i = 0; i< capacity; i++){
    t.data[i] = (T)rand()/(T)RAND_MAX;
  }
}

void _tensor_fill_patterned(tensor_t t){
  uint capacity = dim_get_capacity(t.dim);
  uint i;
  for(i = 0; i < capacity; i++){
    t.data[i] = (T)(i);
  }
}

void tensor_fill_list(tensor_t const t, T const value_list[],
                        uint const length_of_value_list){
  assert(length_of_value_list <= tensor_get_capacity(t));
  for(uint i = 0; i< length_of_value_list; i ++){
    t.data[i] = value_list[i];
  }
}

// TODO : add error handling
tensor_t _tensor_make(dim_t dim){
  tensor_t t;
  uint capacity;
  capacity = dim_get_capacity(dim);
  t.data = malloc(capacity * sizeof(T));
  t.mem_type = CPU_MEM;
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

  for(i = 0; i < MAX_DIM; i++){
    if(i < ndims)
      dim.dims[i] = shape[i];
    else
      dim.dims[i] = 0;
  }
  return _tensor_make(dim);
}

tensor_t tensor_make_empty_with_dim(dim_t dim){
  tensor_t empty;
  empty.dim = dim;
  empty.data = NULL;

  return empty;
}

tensor_t tensor_make_random(uint const shape[], uint const ndims, int seed){
  tensor_t t = tensor_make(shape, ndims);
  _tensor_fill_random(t, seed);
  return t;
}

tensor_t tensor_make_copy(tensor_t t){
  tensor_t ret = _tensor_make(t.dim);
  tensor_copy(ret, t);
  return ret;
}

tensor_t tensor_make_alike(tensor_t t){
  return _tensor_make(t.dim);
}

tensor_t tensor_make_transpose(tensor_t const t){
  uint i,j;
  if(tensor_get_ndims(t) != 2){
    PERR("currently only support 2d transpose")
    tensor_t ret;
    ret.mem_type = BAD_MEM;
    return ret;
  }
  uint M = t.dim.dims[0];
  uint N = t.dim.dims[1];

  dim_t tranposed_dim = dim_get_reverse(t.dim);
  tensor_t t_transposed = _tensor_make(tranposed_dim);

  for(i = 0; i< N; i++){
    for(j = 0; j< M; j++){
      *tensor_get_elem_ptr(t_transposed, make_dim(2, i,j)) = *tensor_get_elem_ptr(t, make_dim(2, j, i));
    }
  }
  return t_transposed;
}

/* TODO @brief fill a tensor with single scalar*/
static void _tensor_fill_scalar(tensor_t t, T s) {
  uint capacity = tensor_get_capacity(t);
  for(uint i=0;i< capacity; i++)
    t.data[i] = s;
}


tensor_t tensor_make_scalar_alike(tensor_t t, T scalar){
  tensor_t tmp = _tensor_make(t.dim);
  _tensor_fill_scalar(tmp, scalar);
  return tmp;
}


tensor_t tensor_make_scalar(uint const shape[], uint const ndims, T s){
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_scalar(t, s);
  return t;
}

tensor_t tensor_make_sum(tensor_t const t, uint const axis_id){
  assert(axis_id ==0); // TODO: currently only support sum along the first dim
  dim_t new_dim = t.dim;
  uint nr_slices = new_dim.dims[axis_id];
  new_dim.dims[axis_id] = 1;

  tensor_t t_ret = _tensor_make(new_dim);
  _tensor_fill_scalar(t_ret, 0.0);

  uint slice_capacity = tensor_get_capacity(t)/nr_slices;

  for( uint i = 0; i< nr_slices; i++ ){
    _add(t_ret.data, t.data + i*slice_capacity, slice_capacity);
  }
  return t_ret;
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
  return S_OK;
}

tensor_t tensor_make_linspace(T const start, T const stop, uint const shape[], uint const ndims){
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_linspace(t, start, stop);
  return t;
}

tensor_t tensor_make_zeros(uint const shape[], uint const ndims) {
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_scalar(t, 0.0);
  return t;
}
tensor_t tensor_make_ones(uint const shape[], uint const ndims) {
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_scalar(t, 1.0);
  return t;
}

tensor_t tensor_make_linspace_alike(T const start, T const stop, tensor_t const t){
  tensor_t ret = _tensor_make(t.dim);
  _tensor_fill_linspace(ret, start, stop);
  return ret;
}

tensor_t tensor_make_patterned(uint const shape[], uint const ndims){
  tensor_t t =  tensor_make(shape, ndims);
  _tensor_fill_patterned(t);
  return t;
}

// p is pad size
tensor_t tensor_make_padded_square_input(tensor_t t, uint p, float pad_val) {
  uint N, C, H, W, HH, WW;
  N = t.dim.dims[0];
  C = t.dim.dims[1];
  H = t.dim.dims[2];
  W = t.dim.dims[3];
  HH = H + 2 * p;
  WW = W + 2 * p;

  uint new_shape[] = {N, C, HH, WW};

  tensor_t n = tensor_make(new_shape, 4);  // 4 is the number of dimensions... TODO fix this
  for (int i = 0; i < N; i++)
    for (int j = 0; j < C; j++)
      for(int k = 0; k < HH; k++)
        for(int l = 0; l < WW; l++) {
          uint target_idx = i * C * HH * WW + j * HH * WW + k * WW + l;
          if (k < p) {
            n.data[target_idx] = pad_val;
          } else if (k >= (H + p)) {
            n.data[target_idx] = pad_val;
          } else if (l < p) {
            n.data[target_idx] = pad_val;
          } else if (l >= (W + p)) {
            n.data[target_idx] = pad_val;
          } else {
            uint src_idx = i * C * H * W + j * H * W + (k - p) * W + (l - p);
            n.data[target_idx] = t.data[src_idx];
          }
        }

  return n;
}

T* tensor_get_elem_ptr(tensor_t const t, dim_t const loc) {
  uint index_dim;
  uint ndims = tensor_get_ndims(t);
  uint offset = 0;
  for(index_dim = 0; index_dim < ndims; index_dim ++){
    offset += loc.dims[index_dim];
    if(index_dim < ndims -1) {
      offset *= t.dim.dims[index_dim + 1];
    }
  }
  // PINF("offset =  %u", offset);
  return t.data + offset;
}

static void _dump(T* data, dim_t dim, int cur_dim_id, int cur_capacity){
  uint i;
  for (i = 0; i < dim.dims[cur_dim_id]; i++){
    if(cur_dim_id + 1 == dim_get_ndims(dim)){ // this is the vector
      PSTR("%.7f ", data[i]);
    }
    else{
      PSTR("{");
      _dump(data + i * (cur_capacity), dim, cur_dim_id + 1, cur_capacity / dim.dims[cur_dim_id + 1]);
      PSTR("}\n");
    }

  }
}

void tensor_dump(tensor_t t){
  PINF("\n$$Dump tensor:");
  dim_t dim = t.dim;
  dim_dump(t.dim);
  uint capacity = dim_get_capacity(dim);
  PSTR("{");
  if(dim.dims[0] == 0)
    PSTR("%.3f ", t.data[0]); //scalar
  else{
    _dump(t.data, dim, 0, capacity / dim.dims[0]);
  }
  PSTR("}\n");
}

// np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
T tensor_rel_error(tensor_t x, tensor_t ref){
  if(S_OK !=  dim_is_same(x.dim, ref.dim)){
    PERR("Dimension not match!");
    return -1;
  }
  uint capacity = tensor_get_capacity(x);
  T norm_diff = 0; // l-2 norm of difference
  T norm_ref = 0;  // l-2 norm of reference
  for(uint i = 0; i < capacity; i++){
    register T a, r;
    a = x.data[i];
    r = ref.data[i];
    norm_diff += (a - r) * (a - r);
    norm_ref += (r * r);
  }
  assert(norm_ref > 0);
  return norm_diff / norm_ref;
}

void tensor_destroy(tensor_t t){
  if(t.data){
    free(t.data);
    t.data = NULL;
  }
}


