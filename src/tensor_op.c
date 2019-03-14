#include "awnn/tensor.h"
#include "awnn/logging.h"

#include "cblas.h"
#include <assert.h>

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

  // https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      1, in1.data, k,
      in2.data, n,
      1.0, out.data, n);
  ret = S_OK;

end:
  return ret;
}
