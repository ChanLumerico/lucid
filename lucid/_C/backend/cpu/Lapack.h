// lucid/_C/backend/cpu/Lapack.h
//
// Wrappers around Apple Accelerate LAPACK routines that handle the row-major ↔
// column-major layout conversion required by all LAPACK calls.  LAPACK expects
// column-major ("Fortran order") matrices; Lucid stores row-major tensors, so
// every wrapper transposes the input before the LAPACK call and transposes the
// output back to row-major before returning.  Transpose is performed using
// vDSP_mtrans for zero-copy efficiency.
//
// All functions write a LAPACK info code into *info: 0 = success,
// negative = illegal argument, positive = algorithmic failure (singular
// matrix, convergence failure, etc.).

#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// LU-based matrix inversion: A is overwritten with A^{-1}.
// Uses sgetrf_ / sgetri_ (float) or dgetrf_ / dgetri_ (double).
LUCID_INTERNAL void lapack_inv_f32(float* A, int n, int* info);
LUCID_INTERNAL void lapack_inv_f64(double* A, int n, int* info);

// Solves the linear system AX = B; B is overwritten with the solution X.
// nrhs is the number of right-hand-side columns.
// Uses sgesv_ / dgesv_.
LUCID_INTERNAL void lapack_solve_f32(float* A, float* B, int n, int nrhs, int* info);
LUCID_INTERNAL void lapack_solve_f64(double* A, double* B, int n, int nrhs, int* info);

// LU decomposition: writes the L (unit-lower) and U (upper) factors into
// L_out and U_out respectively.  ipiv receives the pivot permutation.
// Uses sgetrf_ / dgetrf_.
LUCID_INTERNAL void
lapack_lu_f32(const float* A, int n, int* ipiv, float* L_out, float* U_out, int* info);
LUCID_INTERNAL void
lapack_lu_f64(const double* A, int n, int* ipiv, double* L_out, double* U_out, int* info);

// Cholesky decomposition: A is overwritten with the Cholesky factor.
// lower=true produces the lower triangular factor L (A = L L^T);
// lower=false produces the upper triangular factor U (A = U^T U).
// After the call the off-triangle entries are zeroed.
// Uses spotrf_ / dpotrf_.
LUCID_INTERNAL void lapack_cholesky_f32(float* A, int n, bool lower, int* info);
LUCID_INTERNAL void lapack_cholesky_f64(double* A, int n, bool lower, int* info);

// Reduced QR decomposition of an m×n matrix.
// Q is written as an m×k matrix (k = min(m,n)) and R as a k×n matrix.
// Uses sgeqrf_ + sorgqr_ / dgeqrf_ + dorgqr_.
LUCID_INTERNAL void lapack_qr_f32(const float* A, int m, int n, float* Q, float* R, int* info);
LUCID_INTERNAL void lapack_qr_f64(const double* A, int m, int n, double* Q, double* R, int* info);

// Singular value decomposition.  full_matrices=true uses economy (thin) SVD
// when false, but the parameter name mirrors NumPy's convention.
// U is m×u_cols, S is k-element, Vt is vt_rows×n (row-major).
// Uses sgesdd_ / dgesdd_ (divide-and-conquer, faster than sgesvd_).
LUCID_INTERNAL void lapack_svd_f32(
    const float* A, int m, int n, bool full_matrices, float* U, float* S, float* Vt, int* info);
LUCID_INTERNAL void lapack_svd_f64(
    const double* A, int m, int n, bool full_matrices, double* U, double* S, double* Vt, int* info);

// Symmetric eigendecomposition (real, symmetric input).  Eigenvalues are
// written to w in ascending order; eigenvectors are the columns of V_out
// (stored row-major after transposition).
// Uses ssyevd_ / dsyevd_ (divide-and-conquer driver).
LUCID_INTERNAL void lapack_eigh_f32(const float* A, int n, float* w, float* V_out, int* info);
LUCID_INTERNAL void lapack_eigh_f64(const double* A, int n, double* w, double* V_out, int* info);

// General (non-symmetric) eigendecomposition.  Real and imaginary parts of
// eigenvalues are split into wr and wi.  VR is the right-eigenvector matrix
// (may be null if eigenvectors are not needed).
// Uses sgeev_ / dgeev_.
LUCID_INTERNAL void
lapack_eig_f32(const float* A, int n, float* wr, float* wi, float* VR, int* info);
LUCID_INTERNAL void
lapack_eig_f64(const double* A, int n, double* wr, double* wi, double* VR, int* info);

// LU factorisation (packed format matching LAPACK dgetrf_ output).
// LU_out receives the packed LU matrix (n×n, row-major); ipiv_out receives
// the 1-based pivot indices (n int32_t values).
// Uses sgetrf_ / dgetrf_.
LUCID_INTERNAL void lapack_lu_factor_f32(const float* A, int n, float* LU_out,
                                         int* ipiv_out, int* info);
LUCID_INTERNAL void lapack_lu_factor_f64(const double* A, int n, double* LU_out,
                                         int* ipiv_out, int* info);

// Triangular solve: solve A X = B where A is triangular; B is overwritten
// with X.  upper selects upper (true) or lower (false) triangular; unit
// indicates a unit-diagonal (implicit 1 on diagonal).
// Uses strtrs_ / dtrtrs_.
LUCID_INTERNAL void lapack_solve_triangular_f32(const float* A, float* B, int n,
                                                int nrhs, bool upper, bool unit,
                                                int* info);
LUCID_INTERNAL void lapack_solve_triangular_f64(const double* A, double* B, int n,
                                                int nrhs, bool upper, bool unit,
                                                int* info);

// Least-squares: min ||AX - B||_2 using sgels_/dgels_.
// A is m×n (col-major on input, overwritten); B is m×nrhs, overwritten with X.
LUCID_INTERNAL void lapack_lstsq_f32(const float* A, float* B, int m, int n, int nrhs, int* info);
LUCID_INTERNAL void lapack_lstsq_f64(const double* A, double* B, int m, int n, int nrhs, int* info);

// LU-based solve: AX=B given packed LU+ipiv from lu_factor. Uses sgetrs_/dgetrs_.
// B is overwritten with X.
LUCID_INTERNAL void lapack_lu_solve_f32(const float* LU, const int* ipiv,
                                        float* B, int n, int nrhs, int* info);
LUCID_INTERNAL void lapack_lu_solve_f64(const double* LU, const int* ipiv,
                                        double* B, int n, int nrhs, int* info);

// Householder product: reconstruct Q from H (geqrf output) + tau. Uses sorgqr_/dorgqr_.
// Writes Q (m×k) into Q_out.
LUCID_INTERNAL void lapack_householder_product_f32(const float* H, const float* tau,
                                                   float* Q_out, int m, int n, int k, int* info);
LUCID_INTERNAL void lapack_householder_product_f64(const double* H, const double* tau,
                                                   double* Q_out, int m, int n, int k, int* info);

// LDL^T factorization. Uses ssytrf_/dsytrf_. A_out receives packed LD; ipiv receives pivots.
LUCID_INTERNAL void lapack_ldl_factor_f32(const float* A, float* A_out,
                                          int* ipiv, int n, int* info);
LUCID_INTERNAL void lapack_ldl_factor_f64(const double* A, double* A_out,
                                          int* ipiv, int n, int* info);

// Low-level helpers: convert a row-major matrix to column-major (for LAPACK
// input) and back (for LAPACK output) using vDSP_mtrans.
LUCID_INTERNAL void transpose_to_col_major_f32(const float* src, float* dst, int rows, int cols);
LUCID_INTERNAL void transpose_to_col_major_f64(const double* src, double* dst, int rows, int cols);
LUCID_INTERNAL void transpose_from_col_major_f32(const float* src, float* dst, int rows, int cols);
LUCID_INTERNAL void
transpose_from_col_major_f64(const double* src, double* dst, int rows, int cols);

}  // namespace lucid::backend::cpu
