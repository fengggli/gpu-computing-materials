#include "awnndevice/cublas_wrappers.cuh"


//// transpose call double
static inline cublasStatus_t cublasTgeam(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n,
                                  const double *alpha,
                                  const double *A, int lda,
                                  const double *beta,
                                  const double *B, int ldb,
                                  double *C, int ldc)
{
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
                     beta, B, ldb, C, ldc);
}


static cublasStatus_t inline cublasTgeam(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n,
                                  const float *alpha,
                                  const float *A, int lda,
                                  const float *beta,
                                  const float *B, int ldb,
                                  float *C, int ldc)
{
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
                     beta, B, ldb, C, ldc);
}



tensor_t transpose_device (cublasHandle_t handle, tensor_t src)
{
  assert(src.mem_type == GPU_MEM);

  uint trans_shape[] = { src.dim.dims[1], src.dim.dims[0] };
  tensor_t trans = tensor_make_device(trans_shape, ARRAY_SIZE(trans_shape));

  T const alpha(1.0);
  T const beta(0.0);

  int m = src.dim.dims[0];
  int n = src.dim.dims[1];

  cublasTgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
              m, n, &alpha,
              src.data, n, &beta,
              src.data, m,
              trans.data, m
  );

  return trans;
}


static cublasStatus_t cublasTgemm(cublasHandle_t handle,
                                         cublasOperation_t op_a,
                                         cublasOperation_t op_b,
                                         int m, int n, int k, const float * alpha,
                                         const float *A, int lda, const float *B,
                                         int ldb, const float * beta, float *C,
                                         int ldc)
{
  return cublasSgemm(handle, op_a, op_b, m, n, k, alpha, A,
                     lda, B, ldb, beta, C, ldc);
}


static cublasStatus_t cublasTgemm(cublasHandle_t handle,
                                         cublasOperation_t op_a,
                                         cublasOperation_t op_b,
                                         int m, int n, int k, const double * alpha,
                                         const double *A, int lda, const double *B,
                                         int ldb, const double * beta, double *C,
                                         int ldc)
{
  return cublasDgemm(handle, op_a, op_b, m, n, k, alpha, A,
                     lda, B, ldb, beta, C, ldc);
}

tensor_t cublas_gemm_launch(cublasHandle_t handle, tensor_t d_A, tensor_t d_B) {
//  print_tensor_device<<<1, 1>>>(d_A);
//  print_tensor_device<<<1, 1>>>(d_B);

  const T alpha = 1.f;
  const T beta = 0.f;

  const int rowA = d_A.dim.dims[0];
  const int colA = d_A.dim.dims[1];
  const int colB = d_B.dim.dims[1];

  const T * srcA  = d_A.data;
  const T * srcB  = d_B.data;

  uint shape_res[] = { (uint)rowA, (uint)colB };
  tensor_t result = tensor_make_device(shape_res, ARRAY_SIZE(shape_res));
  T * out   = result.data;

  // Do the actual multiplication
  cublasTgemm(handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              colB,
              rowA,
              colA,
              &alpha,
              srcB,
              colB,
              srcA,
              colA,
              &beta,
              out,
              colB);

//  print_tensor_device<<<1, 1>>>(result);
  return result;
}