#pragma once

// =====================================================================
// Lucid C++ engine — Accelerate LAPACK wrappers.
// =====================================================================
//
// Apple Accelerate ships LAPACK alongside BLAS. The Fortran ABI uses
// column-major storage and trailing-underscore function names. These
// wrappers transpose row-major Lucid tensors to column-major before each
// LAPACK call and back on return, so callers can stay row-major end-to-end.
//
// Each wrapper handles ONE matrix of size (m, n). Batched dispatch is the
// caller's responsibility — loop over leading dims and pass each slice.
//
// All routines are F32 / F64 only (the only float dtypes Lucid supports).
// `info` follows the LAPACK convention: 0 == success; <0 == invalid arg
// (negative of the offending arg index); >0 == math failure (e.g. singular
// for getri, non-PD for potrf, no convergence for gesdd).
//
// Layer: backend/cpu/. Used only by lucid/_C/ops/linalg/.

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// ----- LU + inverse + solve --------------------------------------------- //
//
// All matrices are passed as row-major. The wrappers handle transposition
// internally; LAPACK's own ipiv buffer is the caller's responsibility
// (size n).

LUCID_INTERNAL void lapack_inv_f32(float* A, int n, int* info);
LUCID_INTERNAL void lapack_inv_f64(double* A, int n, int* info);

// Solve A X = B in-place. A is (n,n), B is (n,nrhs); on return B holds X.
LUCID_INTERNAL void lapack_solve_f32(float* A, float* B, int n, int nrhs, int* info);
LUCID_INTERNAL void lapack_solve_f64(double* A, double* B, int n, int nrhs, int* info);

// LU factorize A (n,n). On return:
//   ipiv    : 1-based row swaps (LAPACK convention; size n)
//   L_out   : (n,n) row-major lower-triangular with unit diagonal
//   U_out   : (n,n) row-major upper-triangular
// Used by det_op for the (sign × prod(diag U)) form.
LUCID_INTERNAL void lapack_lu_f32(const float* A, int n, int* ipiv,
                                   float* L_out, float* U_out, int* info);
LUCID_INTERNAL void lapack_lu_f64(const double* A, int n, int* ipiv,
                                   double* L_out, double* U_out, int* info);

// ----- Cholesky --------------------------------------------------------- //
//
// On return, A is overwritten so that:
//   lower=true  → A is L (lower triangular), strict-upper zeroed.
//   lower=false → A is U (upper triangular), strict-lower zeroed.

LUCID_INTERNAL void lapack_cholesky_f32(float* A, int n, bool lower, int* info);
LUCID_INTERNAL void lapack_cholesky_f64(double* A, int n, bool lower, int* info);

// ----- QR --------------------------------------------------------------- //
//
// "Reduced" QR: A is (m, n). Outputs:
//   Q : (m, k) row-major, k = min(m,n), columns orthonormal
//   R : (k, n) row-major, upper-triangular

LUCID_INTERNAL void lapack_qr_f32(const float* A, int m, int n,
                                   float* Q, float* R, int* info);
LUCID_INTERNAL void lapack_qr_f64(const double* A, int m, int n,
                                   double* Q, double* R, int* info);

// ----- SVD -------------------------------------------------------------- //
//
// A is (m, n). Outputs (full_matrices=false → reduced; true → full):
//   U  : (m, k) reduced or (m, m) full   — k = min(m,n)
//   S  : (k,) singular values
//   Vt : (k, n) reduced or (n, n) full   — Vh = V^H (already transposed)

LUCID_INTERNAL void lapack_svd_f32(const float* A, int m, int n,
                                    bool full_matrices,
                                    float* U, float* S, float* Vt, int* info);
LUCID_INTERNAL void lapack_svd_f64(const double* A, int m, int n,
                                    bool full_matrices,
                                    double* U, double* S, double* Vt, int* info);

// ----- Eigenvalues ------------------------------------------------------ //
//
// Symmetric / Hermitian: real eigenvalues, orthogonal eigenvectors.
//   A    : (n, n) row-major. Only the lower triangle is read (LAPACK 'L').
//   w    : (n,) eigenvalues in ascending order.
//   V_out: (n, n) row-major; columns are normalized eigenvectors.

LUCID_INTERNAL void lapack_eigh_f32(const float* A, int n, float* w,
                                     float* V_out, int* info);
LUCID_INTERNAL void lapack_eigh_f64(const double* A, int n, double* w,
                                     double* V_out, int* info);

// General (non-symmetric): complex eigenpairs returned as (real, imag) parts.
//   A     : (n,n) row-major.
//   wr/wi : (n,) real / imag parts of eigenvalues.
//   VR    : (n,n) right eigenvectors (real Schur form — see LAPACK docs);
//           may be nullptr if not needed.

LUCID_INTERNAL void lapack_eig_f32(const float* A, int n,
                                    float* wr, float* wi,
                                    float* VR, int* info);
LUCID_INTERNAL void lapack_eig_f64(const double* A, int n,
                                    double* wr, double* wi,
                                    double* VR, int* info);

// ----- Internal layout helpers (exposed for ops that batch manually) ---- //

LUCID_INTERNAL void transpose_to_col_major_f32(const float* src, float* dst,
                                                int rows, int cols);
LUCID_INTERNAL void transpose_to_col_major_f64(const double* src, double* dst,
                                                int rows, int cols);
LUCID_INTERNAL void transpose_from_col_major_f32(const float* src, float* dst,
                                                  int rows, int cols);
LUCID_INTERNAL void transpose_from_col_major_f64(const double* src, double* dst,
                                                  int rows, int cols);

}  // namespace lucid::backend::cpu
