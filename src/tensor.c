#define CONFIG_DEBUG

#include "awnn/logging.h"
#include "awnn/memory.h"
#include "awnn/tensor.h"

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
  for (i = 0; i < MAX_DIM; i++) {
    if (i < ndims)
      dim.dims[i] = va_arg(vl, uint);
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
  for (i = 0; i < ndims; i++) {
    ret_dim.dims[ndims - i - 1] = dim.dims[i];
  }
  for (i = ndims; i < MAX_DIM; i++) ret_dim.dims[i] = 0;
  return ret_dim;
}

uint dim_get_capacity(dim_t dim) {
  int i;
  uint size = 1;
  for (i = 0; i < MAX_DIM; i++) {
    uint tmp = dim.dims[i];
    if (tmp > 0)
      size *= tmp;
    else
      break;
  }
  return size;
}

uint dim_get_ndims(dim_t dim) {
  int i;
  uint ndims = 0;
  for (i = 0; i < MAX_DIM; i++) {
    uint tmp = dim.dims[i];
    if (tmp > 0)
      ndims++;
    else
      break;
  }
  return ndims;
}

status_t dim_is_same(dim_t dim1, dim_t dim2) {
  uint i;
  for (i = 0; i < MAX_DIM; i++) {
    if (dim1.dims[i] != dim2.dims[i]) {
      return S_BAD_DIM;
    }
  }
  return S_OK;
}

status_t dim_is_capable(dim_t dim1, dim_t dim2) {
  uint i;
  if(dim1.dims[0] < dim2.dims[0]) return S_BAD_DIM;
  for (i = 1; i < MAX_DIM; i++) {
    if (dim1.dims[i] != dim2.dims[i]) {
      return S_BAD_DIM;
    }
  }
  return S_OK;
}

void dim_dump(dim_t dim) {
  int i;
  PSTR("Dimension Dump: [");
  for (i = 0; i < MAX_DIM; i++) {
    uint tmp = dim.dims[i];
    if (tmp > 0)
      PSTR("%d ", tmp);
    else
      break;
  }
  PSTR("]\n");
}

/*
 * Tensor
 */

T tensor_get_sum(tensor_t t) {
  T ret = 0;
  for (uint i = 0; i < tensor_get_capacity(t); i++) {
    ret += t.data[i];
  }
  return ret;
}

void tensor_fill_random(tensor_t t, uint seed) {
  srand(seed);
  uint capacity = dim_get_capacity(t.dim);
  uint i;
  for (i = 0; i < capacity; i++) {
    t.data[i] = (T)rand() / (T)RAND_MAX;
  }
}

void tensor_fill_random_uniform(tensor_t t, double const low, double const high,
                                uint seed) {
  assert(high > low);
  srand(seed);
  uint capacity = dim_get_capacity(t.dim);
  uint i;
  for (i = 0; i < capacity; i++) {
    double scale = (T)rand() / (T)RAND_MAX;  // 0~1
    t.data[i] = (T)(low + (high - low) * scale);
  }
}

void tensor_fill_patterned(tensor_t t) {
  uint capacity = dim_get_capacity(t.dim);
  uint i;

  for (i = 0; i < capacity; i++) {
    t.data[i] = (T)(i);
  }
}

void tensor_fill_list(tensor_t const t, double const value_list[],
                      uint const length_of_value_list) {
  assert(length_of_value_list <= tensor_get_capacity(t));
  for (uint i = 0; i < length_of_value_list; i++) {
    t.data[i] = (T)(value_list[i]);
  }
}

// TODO : add error handling
tensor_t _tensor_make(dim_t dim) {
  tensor_t t;
  uint capacity;
  capacity = dim_get_capacity(dim);
  t.data = mem_alloc(capacity * sizeof(T));
  t.mem_type = CPU_MEM;
  t.dim = dim;
  AWNN_CHECK_NE(NULL, t.data);
  return t;
}

tensor_t tensor_make_placeholder(uint const shape[], uint const ndims){
  uint i;
  dim_t dim;

  if (ndims == 0) {
    PINF("placeholder: make zero");
    dim = make_dim(0, 0);
  }

  for (i = 0; i < MAX_DIM; i++) {
    if (i < ndims)
      dim.dims[i] = shape[i];
    else
      dim.dims[i] = 0;
  }
  tensor_t ret;
  ret.dim = dim;
  ret.mem_type = EMPTY_MEM;
  ret.data = NULL;
  return ret;
}

tensor_t tensor_make(uint const shape[], uint const ndims) {
  uint i;
  dim_t dim;

  if (ndims == 0) {
    PINF("make zero, ndims = 0!");
    dim = make_dim(0, 0);
  }

  for (i = 0; i < MAX_DIM; i++) {
    if (i < ndims)
      dim.dims[i] = shape[i];
    else
      dim.dims[i] = 0;
  }
  return _tensor_make(dim);
}

tensor_t tensor_make_empty_with_dim(dim_t dim) {
  tensor_t empty;
  empty.dim = dim;
  empty.data = NULL;

  return empty;
}

tensor_t tensor_make_random(uint const shape[], uint const ndims, uint seed) {
  tensor_t t = tensor_make(shape, ndims);
  tensor_fill_random(t, seed);
  return t;
}

tensor_t tensor_make_copy(tensor_t t) {
  tensor_t ret = _tensor_make(t.dim);
  tensor_copy(ret, t);
  return ret;
}

// TODO : refactor to reflect host creation / mem_type, etc...
tensor_t tensor_make_alike(tensor_t t) { 
  AWNN_CHECK_EQ(t.mem_type, CPU_MEM);
  return _tensor_make(t.dim); 
}

tensor_t tensor_make_zeros_alike(tensor_t t) {
  tensor_t ret = _tensor_make(t.dim);
  tensor_fill_scalar(ret, 0.0);
  return ret;
}

tensor_t tensor_make_transpose(tensor_t const t) {
  uint i, j;
  if (tensor_get_ndims(t) != 2) {
    PERR("currently only support 2d transpose")
    tensor_t ret;
    ret.mem_type = BAD_MEM;
    return ret;
  }
  uint M = t.dim.dims[0];
  uint N = t.dim.dims[1];

  dim_t tranposed_dim;
  tranposed_dim.dims[0] = N;
  tranposed_dim.dims[1] = M;
  tranposed_dim.dims[2] = 0;
  tranposed_dim.dims[3] = 0;
  tensor_t t_transposed = _tensor_make(tranposed_dim);

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      t_transposed.data[i*M +j] = t.data[j*N+i];
    }
  }
  return t_transposed;
}

/* TODO @brief fill a tensor with single scalar*/
void tensor_fill_scalar(tensor_t t, T s) {
  uint capacity = tensor_get_capacity(t);
  for (uint i = 0; i < capacity; i++) t.data[i] = s;
}

tensor_t tensor_make_scalar_alike(tensor_t t, T scalar) {
  tensor_t tmp = _tensor_make(t.dim);
  tensor_fill_scalar(tmp, scalar);
  return tmp;
}

tensor_t tensor_make_scalar(uint const shape[], uint const ndims, T s) {
  tensor_t t = tensor_make(shape, ndims);
  tensor_fill_scalar(t, s);
  return t;
}

tensor_t tensor_make_sum(tensor_t const t, uint const axis_id) {
  assert(axis_id == 0);  // TODO: currently only support sum along the first dim
  dim_t new_dim = t.dim;
  uint nr_slices = new_dim.dims[axis_id];
  new_dim.dims[axis_id] = 1;

  tensor_t t_ret = _tensor_make(new_dim);
  tensor_fill_scalar(t_ret, 0.0);

  uint slice_capacity = tensor_get_capacity(t) / nr_slices;

  for (uint i = 0; i < nr_slices; i++) {
    _add(t_ret.data, t.data + i * slice_capacity, slice_capacity);
  }
  return t_ret;
}

void tensor_fill_linspace(tensor_t t, double const start, double const stop) {
  uint i;
  uint capacity = dim_get_capacity(t.dim);
  if (stop <= start) {
    PERR("Wrong linspace");
    return;
  }
  double step = (stop - start) / ((T)capacity - 1);
  for (i = 0; i < capacity; i++) {
    t.data[i] = (T)(start + i * step);
  }
}

tensor_t tensor_make_linspace(double const start, double const stop,
                              uint const shape[], uint const ndims) {
  tensor_t t = tensor_make(shape, ndims);
  tensor_fill_linspace(t, start, stop);
  return t;
}

tensor_t tensor_make_zeros(uint const shape[], uint const ndims) {
  tensor_t t = tensor_make(shape, ndims);
  tensor_fill_scalar(t, 0.0);
  return t;
}

tensor_t tensor_make_ones(uint const shape[], uint const ndims) {
  tensor_t t = tensor_make(shape, ndims);
  tensor_fill_scalar(t, 1.0);
  return t;
}

tensor_t tensor_make_linspace_alike(double const start, double const stop,
                                    tensor_t const t) {
  tensor_t ret = _tensor_make(t.dim);
  tensor_fill_linspace(ret, start, stop);
  return ret;
}

tensor_t tensor_make_patterned(uint const shape[], uint const ndims) {
  tensor_t t = tensor_make(shape, ndims);
  tensor_fill_patterned(t);
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

  uint new_shape[] = { N, C, HH, WW };
  tensor_t n = tensor_make(new_shape, ARRAY_SIZE(new_shape));

  for (uint i = 0; i < N; i++)
    for (uint j = 0; j < C; j++)
      for (uint k = 0; k < HH; k++)
        for (uint l = 0; l < WW; l++) {
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

// p is pad size
//tensor_t tensor_make_padded_square_input(tensor_t t, uint p, float pad_val) {
//  uint N, C, H, W, HH, WW;
//  N = t.dim.dims[0];
//  C = t.dim.dims[1];
//  H = t.dim.dims[2];
//  W = t.dim.dims[3];
//  HH = H + 2 * p;
//  WW = W + 2 * p;
//
//  uint new_shape[] = { N, C, HH, WW };
//  tensor_t n = tensor_make(new_shape, ARRAY_SIZE(new_shape));
//
//  uint capacity = tensor_get_capacity(n);
//  uint iter = 0;
//  uint new_img_sz = n.dim.dims[1] * n.dim.dims[2] * n.dim.dims[3];
//  uint channel_sz = n.dim.dims[2] * n.dim.dims[3];
//
//  for (uint i = 0; i < N; i++)
//    for (uint j = 0; j < C; j++)
//      for (uint k = 0; k < HH; k++)
//        for (uint l = 0; l < WW; l++) {
//          uint ii = iter / new_img_sz; // ii is the target image
//          uint jj = (iter / channel_sz) % C; // jj is the channel in the image
//          uint kk = (iter / WW) % HH; // kk is the row in the image
//          uint ll = (iter % WW); // ll is the col in the current image
//          assert(ii == i);
//          assert(jj == j);
//          assert(kk == k);
//          assert(ll == l);
//          uint target_idx = i * C * HH * WW + j * HH * WW + k * WW + l;
//          if (k < p) {
//            n.data[target_idx] = pad_val;
//          } else if (k >= (H + p)) {
//            n.data[target_idx] = pad_val;
//          } else if (l < p) {
//            n.data[target_idx] = pad_val;
//          } else if (l >= (W + p)) {
//            n.data[target_idx] = pad_val;
//          } else {
//            uint src_idx = i * C * H * W + j * H * W + (k - p) * W + (l - p);
//            n.data[target_idx] = t.data[src_idx];
//          }
//          ++iter;
//        }
//
//  return n;
//}

tensor_t tensor_make_remove_padding_square(tensor_t t, uint p) {
  uint N, C, H, W, HH, WW;
  N = t.dim.dims[0];
  C = t.dim.dims[1];
  H = t.dim.dims[2];
  W = t.dim.dims[3];
  HH = H - 2 * p;
  WW = W - 2 * p;

  uint new_shape[] = {N, C, HH, WW};
  tensor_t n = tensor_make(new_shape, ARRAY_SIZE(new_shape));

  for (uint i = 0; i < N; ++i) {
    for (uint j = 0; j < C; ++j) {
      for (uint k = 0; k < HH; ++k) {
        for (uint l = 0; l < WW; ++l) {
          uint target_idx = i * C * HH * WW + j * HH * WW + k * WW + l;
          uint src_idx = i * C * H * W + j * H * W + (k + p) * W + (l + p);
          n.data[target_idx] = t.data[src_idx];
        }
      }
    }
  }

  return n;
}


//tensor_t tensor_make_remove_padding_square(tensor_t t, uint p) {
//  uint N, C, H, W, HH, WW;
//  N = t.dim.dims[0];
//  C = t.dim.dims[1];
//  H = t.dim.dims[2];
//  W = t.dim.dims[3];
//  HH = H - 2 * p;
//  WW = W - 2 * p;
//
//  uint new_shape[] = {N, C, HH, WW};
//  tensor_t n = tensor_make(new_shape, ARRAY_SIZE(new_shape));
//
//  uint capacity = tensor_get_capacity(n);
//  uint iter = 0;
//  uint new_img_sz = n.dim.dims[1] * n.dim.dims[2] * n.dim.dims[3];
//  uint channel_sz = n.dim.dims[2] * n.dim.dims[3];
//
//  for (uint i = 0; i < N; ++i) {
//    for (uint j = 0; j < C; ++j) {
//      for (uint k = 0; k < HH; ++k) {
//        for (uint l = 0; l < WW; ++l) {
//          uint ii = iter / new_img_sz; // ii is the target image
//          uint jj = (iter / channel_sz) % C; // jj is the channel in the image
//          uint kk = (iter / WW) % HH; // kk is the row in the image
//          uint ll = (iter % WW); // ll is the col in the current image
//          assert(ii == i);
//          assert(jj == j);
//          assert(kk == k);
//          assert(ll == l);
//          uint target_idx = i * C * HH * WW + j * HH * WW + k * WW + l;
//          uint src_idx = i * C * H * W + j * H * W + (k + p) * W + (l + p);
//          n.data[target_idx] = t.data[src_idx];
//
//          ++iter;
//        }
//      }
//    }
//  }
//
//  return n;
//}

T* tensor_get_elem_ptr(tensor_t const t, dim_t const loc) {
  uint index_dim;
  uint ndims = tensor_get_ndims(t);
  uint offset = 0;
  for (index_dim = 0; index_dim < ndims; index_dim++) {
    offset += loc.dims[index_dim];
    if (index_dim < ndims - 1) {
      offset *= t.dim.dims[index_dim + 1];
    }
  }
  // PINF("offset =  %u", offset);
  return t.data + offset;
}

static void _dump(T* data, dim_t dim, uint cur_dim_id, uint cur_capacity) {
  uint i;
  for (i = 0; i < dim.dims[cur_dim_id]; i++) {
    if (cur_dim_id + 1 == dim_get_ndims(dim)) {  // this is the vector
      PSTR("%.7f ", data[i]);
    } else {
      PSTR("{");
      _dump(data + i * (cur_capacity), dim, cur_dim_id + 1,
            cur_capacity / dim.dims[cur_dim_id + 1]);
      PSTR("}\n");
    }
  }
}

void tensor_dump(tensor_t t) {
  PINF("\n$$Dump tensor:");
  dim_t dim = t.dim;
  dim_dump(t.dim);
  uint capacity = dim_get_capacity(dim);
  PSTR("{");
  if (dim.dims[0] == 0)
    PSTR("%.3f ", t.data[0]);  // scalar
  else {
    _dump(t.data, dim, 0, capacity / dim.dims[0]);
  }
  PSTR("}\n");
}

// np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
T tensor_rel_error(tensor_t x, tensor_t ref) {
  if (S_OK != dim_is_same(x.dim, ref.dim)) {
    PERR("Dimensions not matched!");
    return 100;
  }
  uint capacity = tensor_get_capacity(x);
  T norm_diff = 0;  // l-2 norm of difference
  T norm_ref = 0;   // l-2 norm of reference
  for (uint i = 0; i < capacity; i++) {
    register T a, r;
    a = x.data[i];
    r = ref.data[i];
    norm_diff += (a - r) * (a - r);
    norm_ref += (r * r);
  }
  if (norm_ref <= 0) {
    PERR("issue detected in tensor_rel_error norm_ref <= 0")
    return 100;
  }
  return norm_diff / norm_ref;
}

void tensor_destroy(tensor_t* t) {
  AWNN_CHECK_NE(t->mem_type, GPU_MEM);
  if (t->mem_type == CPU_MEM) {
    mem_free(t->data);
    t->data = NULL;
  }
}

void dump_tensor_stats(tensor_t t, const char *name) {
  uint capacity = tensor_get_capacity(t);
  double sum = 0;
  for (uint i = 0; i < capacity; i++) {
    sum += t.data[i];
  }
  double mean = sum / capacity;

  sum = 0;
  for (uint i = 0; i < capacity; i++) {
    sum += (t.data[i] - mean) * (t.data[i] - mean);
  }
  double std = sqrt(sum / (capacity - 1));
  PNOTICE("[Tensor(%p) %s]: mean = %.2e, std = %.2e", t.data, name, mean, std);
}
