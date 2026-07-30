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

// Single-precision general matrix multiply (GEMM).
//
// Computes $C \leftarrow \alpha (A B) + \beta C$ in row-major layout using
// Accelerate's ``cblas_sgemm``.  Each transpose flag is mapped to the
// corresponding ``CBLAS_TRANSPOSE`` enum (``CblasNoTrans`` / ``CblasTrans``)
// at zero copy cost — the BLAS kernel selects an alternate inner loop for
// transposed operands.
//
// Parameters
// ----------
// transA, transB : bool
//     Whether $A$ or $B$ should be transposed before the multiply.
// M, N, K : int
//     Output is $M \times N$; the contracted dimension is $K$, so
//     $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$
//     (before optional transposition).
// alpha, beta : float
//     Linear-combination coefficients.  Use $\alpha = 1, \beta = 0$ for a
//     pure multiply; nonzero $\beta$ enables fused accumulate-into-C.
// A, B : const float*
//     Row-major operand buffers.
// C : float*
//     Row-major output buffer; updated in place when $\beta \neq 0$.
// lda, ldb, ldc : int
//     Leading dimensions (row stride in elements) of $A$, $B$, $C$.
//
// Math
// ----
// $$ C_{ij} \leftarrow \alpha \sum_{k=0}^{K-1} A_{ik} B_{kj} + \beta C_{ij} $$
//
// Notes
// -----
// Single-threaded on Apple Silicon for small/medium sizes; Accelerate
// dispatches to AMX (matrix coprocessor) for large GEMMs automatically.
//
// References
// ----------
// BLAS Reference (Netlib), Accelerate.framework ``cblas_sgemm``.
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

// Double-precision general matrix multiply (GEMM).
//
// Identical contract to ``sgemm`` but operates on ``double`` buffers and
// dispatches to ``cblas_dgemm``.
//
// Parameters
// ----------
// transA, transB : bool
//     Whether $A$ or $B$ should be transposed before the multiply.
// M, N, K : int
//     $C \in \mathbb{R}^{M \times N}$, $A \in \mathbb{R}^{M \times K}$,
//     $B \in \mathbb{R}^{K \times N}$ before optional transposition.
// alpha, beta : double
//     Linear-combination coefficients.
// A, B : const double*
//     Row-major operand buffers.
// C : double*
//     Row-major output buffer.
// lda, ldb, ldc : int
//     Leading dimensions.
//
// Math
// ----
// $$ C_{ij} \leftarrow \alpha \sum_{k=0}^{K-1} A_{ik} B_{kj} + \beta C_{ij} $$
//
// References
// ----------
// Accelerate.framework ``cblas_dgemm``.
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

// Single-precision general matrix-vector multiply (GEMV).
//
// Computes $y \leftarrow \alpha (A x) + \beta y$ in row-major layout via
// Accelerate's ``cblas_sgemv``.  Significantly cheaper than ``sgemm`` for
// vector right-hand sides because the inner loop stays in cache.
//
// Parameters
// ----------
// transA : bool
//     If true, compute $y \leftarrow \alpha A^T x + \beta y$ instead.
// M, N : int
//     $A \in \mathbb{R}^{M \times N}$.  When ``transA`` is false, $x$ has
//     $N$ elements and $y$ has $M$; reversed when ``transA`` is true.
// alpha, beta : float
//     Linear-combination coefficients.
// A : const float*
//     Row-major matrix buffer.
// x : const float*
//     Input vector buffer (stride ``incx``).
// y : float*
//     Output vector buffer (stride ``incy``); updated in place.
// lda : int
//     Leading dimension (row stride) of $A$.
// incx, incy : int
//     Element stride within the input/output vectors.
//
// Math
// ----
// $$ y_i \leftarrow \alpha \sum_{j=0}^{N-1} A_{ij} x_j + \beta y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_sgemv``.
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

// Double-precision general matrix-vector multiply (GEMV).
//
// Identical contract to ``sgemv`` but for ``double`` buffers; dispatches to
// ``cblas_dgemv``.
//
// Parameters
// ----------
// transA : bool
//     If true, multiply against $A^T$ instead of $A$.
// M, N : int
//     $A \in \mathbb{R}^{M \times N}$.
// alpha, beta : double
//     Linear-combination coefficients.
// A : const double*
//     Row-major matrix buffer.
// x : const double*
//     Input vector buffer.
// y : double*
//     Output vector buffer; updated in place.
// lda : int
//     Leading dimension of $A$.
// incx, incy : int
//     Element strides of $x$ and $y$.
//
// Math
// ----
// $$ y_i \leftarrow \alpha \sum_{j=0}^{N-1} A_{ij} x_j + \beta y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_dgemv``.
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

// Single-precision scaled vector accumulation, $y \leftarrow \alpha x + y$.
//
// The BLAS name for "add a scaled copy of one vector to another", and the
// building block of any linear combination of same-shaped buffers.  Preferred
// over a multiply into a temporary followed by an add: it needs no temporary,
// touches $y$ once per term instead of three times, and Accelerate's kernel
// contracts the multiply and the add into a single rounding.
//
// Parameters
// ----------
// n : int
//     Element count.
// alpha : float
//     Scale applied to $x$.
// x : const float*
//     Vector to accumulate; unchanged.
// y : float*
//     Accumulator, updated in place.
//
// Math
// ----
// $$ y_i \leftarrow \alpha x_i + y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_saxpy``.
LUCID_INTERNAL void saxpy(int n, float alpha, const float* x, float* y);

// Double-precision scaled vector accumulation, $y \leftarrow \alpha x + y$.
//
// Parameters
// ----------
// n : int
//     Element count.
// alpha : double
//     Scale applied to $x$.
// x : const double*
//     Vector to accumulate; unchanged.
// y : double*
//     Accumulator, updated in place.
//
// Math
// ----
// $$ y_i \leftarrow \alpha x_i + y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_daxpy``.
LUCID_INTERNAL void daxpy(int n, double alpha, const double* x, double* y);

}  // namespace lucid::backend::cpu
