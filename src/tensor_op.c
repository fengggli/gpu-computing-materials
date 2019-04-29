#include "awnn/tensor.h"
#include "awnn/logging.h"
#include "utils/debug.h"

#ifdef USE_OPENBLAS
#include "cblas.h"
#endif
#include <assert.h>

// used for forward
tensor_t tensor_make_transpose_3012(tensor_t t) {
  uint target_idx = 0;
  tensor_t tpose = tensor_make_copy(t);
  for (uint i = 0; i < t.dim.dims[3]; ++i) {  // for each of the new dim 0
    for (uint j = 0; j < t.dim.dims[0] * t.dim.dims[1] * t.dim.dims[2]; ++j) {
      tpose.data[target_idx++] = t.data[i + j * t.dim.dims[3]];
    }
  }
  uint const shape[] = { t.dim.dims[3], t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] };
  tensor_reshape_(&tpose, shape, ARRAY_SIZE(shape));
  return tpose;
}

// used for backward
tensor_t tensor_make_transpose_1230(tensor_t t) {
  uint src_idx = 0, target_idx = 0;
  uint original_dim_0 = t.dim.dims[0];
  uint original_dim_1 = t.dim.dims[1];
  uint original_dim_2 = t.dim.dims[2];
  uint original_dim_3 = t.dim.dims[3];

  tensor_t tpose = tensor_make_copy(t);

  for (uint i = 0; i < original_dim_0; ++i) {
    for (uint j = 0; j < original_dim_1 * original_dim_2 * original_dim_3; ++j) {
      target_idx = (i + j * original_dim_0);
      tpose.data[target_idx] = t.data[src_idx++];
    }
  }

  uint const shape[] = { original_dim_1, original_dim_2, original_dim_3, original_dim_0 };
  tensor_reshape_(&tpose, shape, ARRAY_SIZE(shape));
  return tpose;
}

#ifdef USE_OPENBLAS
status_t awnn_gemm(const int TransA,
    const int TransB, const int M, const int N, const int K, const double alpha,
    const T* A, const T* B, const double beta, T*C){
  int lda = (TransA == CblasNoTrans)? K: M;
  int ldb = (TransB == CblasNoTrans)? N: K;
#ifndef AWNN_USE_FLT32
  cblas_dgemm(CblasRowMajor, TransA, TransB,
      M, N, K,
      (double)alpha, A, lda,
      B, ldb,
      (double)beta, C, N);

#else
  cblas_sgemm(CblasRowMajor, (CBLAS_TRANSPOSE)TransA, (CBLAS_TRANSPOSE)TransB,
      M, N, K,
      (float)alpha, A, lda,
      B, ldb,
      (float)beta, C, N);
#endif

  return S_OK;
}
#endif

// TODO:  results not correct
status_t tensor_matmul(tensor_t in1, tensor_t in2, tensor_t out){
  status_t ret = S_ERR;
  if(dim_get_ndims(in1.dim) != 2 || dim_get_ndims(in2.dim)!=2){
    PERR("Dot only accepts 2d tensor as input");
    goto end;
  }
  if (in1.dim.dims[1] != in2.dim.dims[0]){
    PERR("Input dimensions not match:");
    dim_dump(in1.dim);
    dim_dump(in2.dim);
    goto end;
  }
  if (out.dim.dims[0] != in1.dim.dims[0] || out.dim.dims[1] != in2.dim.dims[1]){
    PERR("Out dimensions not match in1, in2, out dims are:");
    dim_dump(in1.dim);
    dim_dump(in2.dim);
    dim_dump(out.dim);
    print_trace();
    goto end;
  }
  int m = (int)in1.dim.dims[0];
  int k = (int)in1.dim.dims[1];
  int n = (int)in2.dim.dims[1];

  // PDBG("mnk = [%u, %u, %u]", m,n,k);
#ifdef USE_OPENBLAS
  // https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
  tensor_fill_scalar(out, 0.0);
  awnn_gemm(CblasNoTrans, CblasNoTrans, m, n, k, 1.0, in1.data, in2.data, 1.0, out.data);
#else
  uint ii, jj, kk; // A[i.j] with B[j,k]
  for(ii = 0; ii < m; ii++){
    for(kk = 0; kk < n; kk++){
      T tmp = 0;
      for(jj = 0; jj < k; jj++){
        tmp += in1.data[ii * k + jj] * in2.data[jj * n + kk];
      }
      out.data[ii * n + kk] = tmp;
    }
  }

#endif
  ret = S_OK;

end:
  return ret;
}

status_t tensor_elemwise_op_inplace(tensor_t to, tensor_t from, tensor_op_t op){
  if(S_OK == dim_is_same(to.dim, from.dim)){
    switch(op){
      case TENSOR_OP_ADD:
        _add(to.data, from.data, dim_get_capacity(to.dim));
        break;
      case TENSOR_OP_SUB:
        _sub(to.data, from.data, dim_get_capacity(to.dim));
        break;
      case TENSOR_OP_MUL:
        _mul(to.data, from.data, dim_get_capacity(to.dim));
        break;
      case TENSOR_OP_DIV:
        _div(to.data, from.data, dim_get_capacity(to.dim));
        break;
      default:
        PERR("unsupported tensor_op_t =%d", op);
        return S_ERR;
    }
    return S_OK;
  }
  else{
    PERR("[tensor plus inplace]: wrong dims ");
    return S_BAD_DIM;
  }
}

T tensor_sum_of_square(tensor_t const t) {
  T ret = 0;
  for (uint i = 0; i < tensor_get_capacity(t); i++) {
    ret += (t.data[i]) * (t.data[i]);
  }
  return ret;
}

status_t tensor_copy(tensor_t out, tensor_t in){
  uint i;
  uint capacity = dim_get_capacity(out.dim);
  if( S_OK == dim_is_same(out.dim, in.dim)){
    for(i = 0; i < capacity; i++){
      out.data[i] = in.data[i];
    }
    return S_OK;
  }
  else{
    PERR("[tensor plus inplace]: wrong dims ");
    return S_BAD_DIM;
  }
}

status_t tensor_add_sameshape(tensor_t in1, tensor_t in2, tensor_t out){
  if( S_OK == dim_is_same(out.dim, in1.dim)){
    tensor_copy(out, in1);
    tensor_elemwise_op_inplace(out, in2, TENSOR_OP_ADD);
    return S_OK;
  }
  else{
    PERR("[tensor plus]: wrong dims");
    return S_BAD_DIM;
  }
}

status_t tensor_add_vector_inplace(tensor_t t, tensor_t v) {
  dim_t old_dim = t.dim;
  if(tensor_get_ndims(v) != 1) {
    PERR("second operator is not a 1-d vector:");
    dim_dump(v.dim);
    print_trace();
    return S_ERR;
  }
  uint v_capacity = v.dim.dims[0];
  uint d1 = 1;
  uint i_dim;
  for(i_dim = 0; i_dim < tensor_get_ndims(t) -1; i_dim++ ){
    d1 *= t.dim.dims[i_dim];
  }
  if(t.dim.dims[i_dim] != v_capacity) {
    PERR("last dimension of tensor doesn't fit the vector");
    print_trace();
    return S_ERR;
  }

  uint const tmp_shape[] = {d1, v_capacity};
  tensor_reshape_(&t, tmp_shape, 2); // reshape to 2d
  for( uint i = 0; i< d1; i++ ){
    _add(t.data + i*v_capacity, v.data, v_capacity);
  }
  t.dim = old_dim;
  return S_OK;
  // v should fit the last dimension
}

status_t tensor_reshape_(tensor_t* ptr_t, uint const shape[], uint const ndims){
  dim_t req_dim;
  uint i;
  if(ndims == 0){
    PINF("make zero");
    req_dim = make_dim(0,0);
  }

  for(i = 0; i< MAX_DIM; i++){
    if(i < ndims){
      req_dim.dims[i] = shape[i];
    }
    else
      req_dim.dims[i] = 0;
  }
  if(dim_get_capacity(req_dim) != dim_get_capacity(ptr_t->dim)){
    PERR("[tensor reshape]: dimension not matched");
    PERR("Original dimension: ");
    dim_dump(ptr_t->dim);
    PERR("requested dimension: ");
    dim_dump(req_dim);
    print_trace();
    return S_BAD_DIM;
  }
  ptr_t->dim = req_dim;
  return S_OK;
}


status_t tensor_reshape_flat_(tensor_t * t) {
  uint capacity = dim_get_capacity(t->dim);
  uint shape[MAX_DIM];
  int i;
  for (i = 0; i < MAX_DIM - 1; ++i) {
    shape[i] = 1;
  }
  shape[i] = capacity;
  tensor_reshape_(t, shape, MAX_DIM);
  return S_OK;
}


void tensor_print_flat(tensor_t t) {
  uint capacity = tensor_get_capacity(t);
  printf("[");
  uint i;
  for (i = 0; i < capacity - 1; ++i) {
    printf("%.10f, ", t.data[i]);
  }
  printf("%.10f]\n", t.data[i]);
}
