#pragma once

// =====================================================================
// Lucid C++ engine — Accelerate BLAS wrappers (matmul / GEMV).
// =====================================================================
//
// Apple Accelerate ships LAPACK + BLAS — on Apple Silicon it uses the AMX
// coprocessor for FP32/FP64 matmul (10-100× faster than naive). All matrices
// are row-major; transpose flags follow CBLAS convention.
//
// Phase 3.0 only declares the wrappers; the matmul op (Phase 3.1 linalg.h)
// is the first consumer.
//
// Layer: backend/cpu/.

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

/// C = alpha * op(A) * op(B) + beta * C   (row-major, F32)
///
/// op(A) is A or A^T per `transA`. Sizes:
///   op(A): M x K
///   op(B): K x N
///   C    : M x N
///
/// `lda`, `ldb`, `ldc` are leading dimensions in row-major (== row stride).
LUCID_INTERNAL void sgemm(bool transA, bool transB, int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc);

LUCID_INTERNAL void dgemm(bool transA, bool transB, int M, int N, int K,
                          double alpha,
                          const double* A, int lda,
                          const double* B, int ldb,
                          double beta,
                          double* C, int ldc);

/// y = alpha * op(A) * x + beta * y  (row-major, F32)
LUCID_INTERNAL void sgemv(bool transA, int M, int N,
                          float alpha,
                          const float* A, int lda,
                          const float* x, int incx,
                          float beta,
                          float* y, int incy);

LUCID_INTERNAL void dgemv(bool transA, int M, int N,
                          double alpha,
                          const double* A, int lda,
                          const double* x, int incx,
                          double beta,
                          double* y, int incy);

}  // namespace lucid::backend::cpu
