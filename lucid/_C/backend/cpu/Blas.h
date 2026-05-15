// lucid/_C/backend/cpu/Blas.h
//
// Thin wrappers around Apple Accelerate CBLAS routines used by the CPU backend
// for matrix multiplication and matrix-vector multiplication.  All functions
// assume row-major storage (CblasRowMajor) and map the bool transpose flags to
// CBLAS_TRANSPOSE constants.  "s" prefix = float32; "d" prefix = float64.

#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// Single-precision general matrix multiply: C = alpha*(A @ B) + beta*C.
// Leading dimensions lda, ldb, ldc are the column strides of A, B, C.
// transA/transB control whether each matrix is transposed before multiplication.
LUCID_INTERNAL void sgemm(bool transA,
                          bool transB,
                          int M,
                          int N,
                          int K,
                          float alpha,
                          const float* A,
                          int lda,
                          const float* B,
                          int ldb,
                          float beta,
                          float* C,
                          int ldc);

// Double-precision general matrix multiply: C = alpha*(A @ B) + beta*C.
LUCID_INTERNAL void dgemm(bool transA,
                          bool transB,
                          int M,
                          int N,
                          int K,
                          double alpha,
                          const double* A,
                          int lda,
                          const double* B,
                          int ldb,
                          double beta,
                          double* C,
                          int ldc);

// Single-precision matrix-vector multiply: y = alpha*(A @ x) + beta*y.
// incx/incy are the strides within the input/output vectors.
LUCID_INTERNAL void sgemv(bool transA,
                          int M,
                          int N,
                          float alpha,
                          const float* A,
                          int lda,
                          const float* x,
                          int incx,
                          float beta,
                          float* y,
                          int incy);

// Double-precision matrix-vector multiply: y = alpha*(A @ x) + beta*y.
LUCID_INTERNAL void dgemv(bool transA,
                          int M,
                          int N,
                          double alpha,
                          const double* A,
                          int lda,
                          const double* x,
                          int incx,
                          double beta,
                          double* y,
                          int incy);

}  // namespace lucid::backend::cpu
