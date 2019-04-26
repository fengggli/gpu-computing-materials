#pragma once

#include "awnn/common.h"
#include "awnn/tensor.h"

#include <cublas_v2.h>

tensor_t cublas_transpose_launch(cublasHandle_t handle, tensor_t T);

tensor_t cublas_gemm_launch(cublasHandle_t handle, tensor_t d_A, tensor_t d_B);