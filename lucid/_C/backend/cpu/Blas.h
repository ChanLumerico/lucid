#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

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
