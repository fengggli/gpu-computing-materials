#include "awnn/tensor.h"
#include "awnn/logging.h"

#include "cblas.h"
#include <assert.h>


// TODO:  results not correct
status_t tensor_dot(tensor_t in1, tensor_t in2, tensor_t out){
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
    goto end;
  }
  int m = in1.dim.dims[0];
  int k = in1.dim.dims[1];
  int n = in2.dim.dims[1];

  PINF("mnk = [%u, %u, %u]", m,n,k);


#ifdef USE_OPENBLAS
  // https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      1, in1.data, k,
      in2.data, n,
      1.0, out.data, n);
#else
  int ii, jj, kk; // A[i.j] with B[j,k]
  for(ii = 0; ii < m; ii++){
    for(kk = 0; kk < n; kk++){
      T tmp =0;
      for(jj = 0; jj < k; jj++){
        tmp+=in1.data[ii*k+jj]*in2.data[jj*n+kk];
      }
      out.data[ii*n+kk] = tmp;
    }
  }

#endif
  ret = S_OK;

end:
  return ret;
}

void _plus(T* to, T* from, uint len){
  uint i;
  for(i = 0; i< len; i++){
    to[i] += from[i];
  }
}

status_t tensor_plus_inplace(tensor_t to, tensor_t from){
  if(S_OK == dim_is_same(to.dim, from.dim)){
    _plus(to.data, from.data, dim_get_capacity(to.dim));
    return S_OK;
  }
  else{
    PERR("[tensor plus inplace]: wrong dims ");
    return S_BAD_DIM;
  }
}

status_t tensor_copy(tensor_t out, tensor_t in){
  uint i;
  uint capacity = dim_get_capacity(out.dim);
  if( S_OK == dim_is_same(out.dim, in.dim)){
    for(i = 0; i< capacity; i++){
      out.data[i] = in.data[i];
    }
    return S_OK;
  }
  else{
    PERR("[tensor plus inplace]: wrong dims ");
    return S_BAD_DIM;
  }
}

status_t tensor_plus(tensor_t in1, tensor_t in2, tensor_t out){
  if( S_OK == dim_is_same(out.dim, in1.dim)){
    tensor_copy(out, in1);
    tensor_plus_inplace(out, in2);
    return S_OK;
  }
  else{
    PERR("[tensor plus]: wrong dims");
    return S_BAD_DIM;
  }
}

status_t tensor_reshape_(tensor_t* ptr_t, uint const  shape[], uint const ndims){
  dim_t req_dim;
  uint i;
  if(ndims == 0){
    PINF("make zero");
    req_dim = make_dim(0,0);
  }

  for(i = 0; i< MAX_DIM; i++){
    if(i< ndims){
      req_dim.dims[i] =shape[i];
    }
    else
      req_dim.dims[i] = 0;
  }
  if(dim_get_capacity(req_dim) != dim_get_capacity(ptr_t->dim)){
    PERR("[tensor reshape]: dimension notmatch");
    return S_BAD_DIM;
  }
  ptr_t->dim = req_dim;
  return S_OK;
}
