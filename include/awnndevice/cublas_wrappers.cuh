#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"

#include <cublas_v2.h>

tensor_t transpose_device(cublasHandle_t handle, tensor_t T);

////  void cublasDot(const T * srcA, const T * srcB, T * out, int rowA, int colA, int colB)
tensor_t cublas_gemm_launch(cublasHandle_t handle, tensor_t d_A, tensor_t d_B);


//inline cublasStatus_t cublasTgemm(cublasHandle_t handle,
//                                  cublasOperation_t op_a,
//                                  cublasOperation_t op_b,
//                                  int m, int n, int k, const float * alpha,
//                                  const float *A, int lda, const float *B,
//                                  int ldb, const float * beta, float *C,
//                                  int ldc);
//
///**
// * cublas<T>gemm() specialization for float
// */
//inline cublasStatus_t cublasTgemm(cublasHandle_t handle,
//                                  cublasOperation_t op_a,
//                                  cublasOperation_t op_b,
//                                  int m, int n, int k, const double *alpha,
//                                  const double *A, int lda, const double *B,
//                                  int ldb, const double * beta, double *C,
//                                  int ldc);
//
//
//inline cublasStatus_t cublasTgeam(cublasHandle_t handle,
//                                  cublasOperation_t transa,
//                                  cublasOperation_t transb,
//                                  int m, int n,
//                                  const double *alpha,
//                                  const double *A, int lda,
//                                  const double *beta,
//                                  const double *B, int ldb,
//                                  double *C, int ldc);
//
//
//inline cublasStatus_t cublasTgeam(cublasHandle_t handle,
//                                  cublasOperation_t transa,
//                                  cublasOperation_t transb,
//                                  int m, int n,
//                                  const float *alpha,
//                                  const float *A, int lda,
//                                  const float *beta,
//                                  const float *B, int ldb,
//                                  float *C, int ldc);
