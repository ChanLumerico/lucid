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

// Single-precision in-place matrix inversion via LU.
//
// Overwrites $A$ with $A^{-1}$.  Internally runs ``sgetrf_`` (LU with
// partial pivoting) followed by ``sgetri_`` (invert using the LU factors).
//
// Parameters
// ----------
// A : float*
//     Row-major $n \times n$ matrix; replaced in place with $A^{-1}$.
// n : int
//     Side length.
// info : int*
//     LAPACK status word.  ``0`` = success; ``< 0`` = illegal argument;
//     ``> 0`` = matrix is exactly singular (no inverse).
//
// Math
// ----
// $$ A A^{-1} = I_n $$
//
// References
// ----------
// LAPACK ``sgetrf_`` + ``sgetri_``.
LUCID_INTERNAL void lapack_inv_f32(float* A, int n, int* info);

// Double-precision in-place matrix inversion via LU.
//
// See ``lapack_inv_f32``; uses ``dgetrf_`` + ``dgetri_``.
//
// Parameters
// ----------
// A : double*
//     Row-major $n \times n$ matrix; replaced with $A^{-1}$.
// n : int
//     Side length.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgetrf_`` + ``dgetri_``.
LUCID_INTERNAL void lapack_inv_f64(double* A, int n, int* info);

// Single-precision linear system solver $A X = B$.
//
// Computes the LU factorization of $A$ (with partial pivoting) and solves
// for $X$ in place of $B$.  Used for square invertible systems.
//
// Parameters
// ----------
// A : float*
//     Row-major $n \times n$ coefficient matrix; overwritten by its LU.
// B : float*
//     Row-major $n \times \text{nrhs}$ right-hand-side; overwritten by $X$.
// n : int
//     Order of $A$.
// nrhs : int
//     Number of right-hand-side columns.
// info : int*
//     LAPACK status (``> 0`` means $A$ is singular at pivot ``info``).
//
// Math
// ----
// $$ A X = B \;\;\Longrightarrow\;\; X = A^{-1} B $$
//
// References
// ----------
// LAPACK ``sgesv_``.
LUCID_INTERNAL void lapack_solve_f32(float* A, float* B, int n, int nrhs, int* info);

// Double-precision linear system solver $A X = B$.  Uses ``dgesv_``.
//
// Parameters
// ----------
// A : double*
//     Row-major $n \times n$ matrix; overwritten by its LU.
// B : double*
//     Row-major $n \times \text{nrhs}$; overwritten by $X$.
// n, nrhs : int
//     System dimensions.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgesv_``.
LUCID_INTERNAL void lapack_solve_f64(double* A, double* B, int n, int nrhs, int* info);

// Single-precision LU decomposition split into separate L and U matrices.
//
// Runs ``sgetrf_`` and then explicitly extracts the unit-lower-triangular
// $L$ and upper-triangular $U$ from the packed LAPACK output.
//
// Parameters
// ----------
// A : const float*
//     Row-major $n \times n$ input.
// n : int
//     Side length.
// ipiv : int*
//     Pivot indices written by ``sgetrf_`` (1-based, length $n$).
// L_out : float*
//     Receives the unit-lower-triangular factor (row-major, $n \times n$).
// U_out : float*
//     Receives the upper-triangular factor (row-major, $n \times n$).
// info : int*
//     LAPACK status word.
//
// Math
// ----
// $$ P A = L U $$ where $P$ is encoded by ``ipiv``.
//
// References
// ----------
// LAPACK ``sgetrf_``.
LUCID_INTERNAL void
lapack_lu_f32(const float* A, int n, int* ipiv, float* L_out, float* U_out, int* info);

// Double-precision LU decomposition split into separate L and U.
// See ``lapack_lu_f32``; uses ``dgetrf_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major $n \times n$ input.
// n : int
//     Side length.
// ipiv : int*
//     1-based pivot indices, length $n$.
// L_out, U_out : double*
//     Receive the $L$ and $U$ factors.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgetrf_``.
LUCID_INTERNAL void
lapack_lu_f64(const double* A, int n, int* ipiv, double* L_out, double* U_out, int* info);

// Single-precision Cholesky decomposition (in place).
//
// Factors a symmetric positive-definite matrix into either $A = L L^T$
// (``lower=true``) or $A = U^T U$ (``lower=false``).  After the LAPACK call
// the off-triangle entries are zeroed so the result is a clean lower / upper
// matrix.
//
// Parameters
// ----------
// A : float*
//     Row-major $n \times n$ symmetric positive-definite matrix; overwritten
//     by the requested triangular factor.
// n : int
//     Side length.
// lower : bool
//     ``true`` for lower triangular $L$, ``false`` for upper triangular $U$.
// info : int*
//     LAPACK status (``> 0`` means $A$ is not positive definite at column
//     ``info``).
//
// Math
// ----
// $$ A = L L^T \quad\text{or}\quad A = U^T U $$
//
// References
// ----------
// LAPACK ``spotrf_``.
LUCID_INTERNAL void lapack_cholesky_f32(float* A, int n, bool lower, int* info);

// Double-precision Cholesky decomposition (in place).  Uses ``dpotrf_``.
//
// Parameters
// ----------
// A : double*
//     Row-major SPD matrix; overwritten by the triangular factor.
// n : int
//     Side length.
// lower : bool
//     Select $L$ (true) or $U$ (false).
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dpotrf_``.
LUCID_INTERNAL void lapack_cholesky_f64(double* A, int n, bool lower, int* info);

// Single-precision reduced (thin) QR decomposition.
//
// Computes $A = Q R$ where $Q$ is $m \times k$ with orthonormal columns and
// $R$ is $k \times n$ upper-triangular, $k = \min(m, n)$.  Internally calls
// ``sgeqrf_`` to produce Householder reflectors then ``sorgqr_`` to
// materialise $Q$.
//
// Parameters
// ----------
// A : const float*
//     Row-major $m \times n$ input.
// m, n : int
//     Matrix dimensions.
// Q : float*
//     Output orthonormal-columns matrix, shape $m \times k$ (row-major).
// R : float*
//     Output upper-triangular matrix, shape $k \times n$ (row-major).
// info : int*
//     LAPACK status word.
//
// Math
// ----
// $$ A = Q R, \;\; Q^T Q = I_k $$
//
// References
// ----------
// LAPACK ``sgeqrf_`` + ``sorgqr_``.
LUCID_INTERNAL void lapack_qr_f32(const float* A, int m, int n, float* Q, float* R, int* info);

// Double-precision reduced QR decomposition.  Uses ``dgeqrf_`` + ``dorgqr_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major $m \times n$ input.
// m, n : int
//     Dimensions.
// Q : double*
//     Output $m \times k$ matrix with orthonormal columns.
// R : double*
//     Output $k \times n$ upper-triangular matrix.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgeqrf_`` + ``dorgqr_``.
LUCID_INTERNAL void lapack_qr_f64(const double* A, int m, int n, double* Q, double* R, int* info);

// Single-precision singular value decomposition via divide-and-conquer.
//
// Computes $A = U \, \mathrm{diag}(S) \, V^T$.  When ``full_matrices`` is
// false the thin SVD is returned: $U \in \mathbb{R}^{m \times k}$,
// $V^T \in \mathbb{R}^{k \times n}$, $k = \min(m, n)$.  When true the full
// matrices ($U \in \mathbb{R}^{m \times m}$, $V^T \in \mathbb{R}^{n \times n}$)
// are returned.
//
// Parameters
// ----------
// A : const float*
//     Row-major $m \times n$ input.
// m, n : int
//     Matrix dimensions.
// full_matrices : bool
//     Controls thin vs full SVD as described above.
// U : float*
//     Output $U$ matrix (row-major).
// S : float*
//     Output singular values in descending order, length $\min(m, n)$.
// Vt : float*
//     Output $V^T$ matrix (row-major).
// info : int*
//     LAPACK status (``> 0`` = SVD did not converge).
//
// Math
// ----
// $$ A = U \, \Sigma \, V^T, \quad \Sigma = \mathrm{diag}(S) $$
//
// References
// ----------
// LAPACK ``sgesdd_`` (divide-and-conquer driver).
LUCID_INTERNAL void lapack_svd_f32(
    const float* A, int m, int n, bool full_matrices, float* U, float* S, float* Vt, int* info);

// Double-precision SVD via divide-and-conquer.  Uses ``dgesdd_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major $m \times n$ input.
// m, n : int
//     Dimensions.
// full_matrices : bool
//     Thin (false) vs full (true) SVD.
// U, S, Vt : double*
//     Output buffers, sized analogously to ``lapack_svd_f32``.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgesdd_``.
LUCID_INTERNAL void lapack_svd_f64(
    const double* A, int m, int n, bool full_matrices, double* U, double* S, double* Vt, int* info);

// Symmetric eigendecomposition (float32) via divide-and-conquer.
//
// For a real symmetric input $A = A^T$, computes the spectral decomposition
// $A = V \, \mathrm{diag}(w) \, V^T$.  Eigenvalues are produced in ascending
// order and eigenvectors are the columns of $V$ (returned row-major after
// transposition).
//
// Parameters
// ----------
// A : const float*
//     Row-major $n \times n$ symmetric matrix.
// n : int
//     Side length.
// w : float*
//     Eigenvalues, length $n$ (ascending).
// V_out : float*
//     Row-major matrix whose columns are the eigenvectors.
// info : int*
//     LAPACK status word.
//
// Math
// ----
// $$ A V = V \, \mathrm{diag}(w) $$
//
// References
// ----------
// LAPACK ``ssyevd_``.
LUCID_INTERNAL void lapack_eigh_f32(const float* A, int n, float* w, float* V_out, int* info);

// Double-precision symmetric eigendecomposition.  Uses ``dsyevd_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major $n \times n$ symmetric matrix.
// n : int
//     Side length.
// w : double*
//     Eigenvalues (ascending).
// V_out : double*
//     Row-major eigenvector matrix.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dsyevd_``.
LUCID_INTERNAL void lapack_eigh_f64(const double* A, int n, double* w, double* V_out, int* info);

// General (non-symmetric) eigendecomposition (float32).
//
// Computes eigenvalues of a general real matrix; complex eigenvalues are
// returned as separate real (``wr``) and imaginary (``wi``) parts.  Right
// eigenvectors (columns of $V$) are written to ``VR`` if non-null.
//
// Parameters
// ----------
// A : const float*
//     Row-major $n \times n$ input.
// n : int
//     Side length.
// wr, wi : float*
//     Real / imaginary parts of the eigenvalues, length $n$.
// VR : float*
//     Right-eigenvector matrix (row-major) or ``nullptr`` to skip vectors.
// info : int*
//     LAPACK status (``> 0`` = QR iteration failed to converge).
//
// Math
// ----
// $$ A v_j = \lambda_j v_j, \quad \lambda_j = \mathrm{wr}_j + i \, \mathrm{wi}_j $$
//
// References
// ----------
// LAPACK ``sgeev_``.
LUCID_INTERNAL void
lapack_eig_f32(const float* A, int n, float* wr, float* wi, float* VR, int* info);

// Double-precision general eigendecomposition.  Uses ``dgeev_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major $n \times n$ input.
// n : int
//     Side length.
// wr, wi : double*
//     Real and imaginary eigenvalue parts.
// VR : double*
//     Right-eigenvector matrix or ``nullptr``.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgeev_``.
LUCID_INTERNAL void
lapack_eig_f64(const double* A, int n, double* wr, double* wi, double* VR, int* info);

// Single-precision LU factorisation (packed format).
//
// Writes the LAPACK-native packed LU output (lower triangle = $L$ without its
// unit diagonal, upper triangle = $U$) into ``LU_out``, along with the
// pivot indices into ``ipiv_out``.  Intended for use with
// ``lapack_lu_solve_f32`` so the factorisation can be reused across many
// right-hand sides.
//
// Parameters
// ----------
// A : const float*
//     Row-major $n \times n$ input.
// n : int
//     Side length.
// LU_out : float*
//     Packed LU output (row-major).
// ipiv_out : int*
//     1-based pivot indices, length $n$.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``sgetrf_``.
LUCID_INTERNAL void
lapack_lu_factor_f32(const float* A, int n, float* LU_out, int* ipiv_out, int* info);

// Double-precision packed LU factorisation.  Uses ``dgetrf_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major $n \times n$ input.
// n : int
//     Side length.
// LU_out : double*
//     Packed LU output.
// ipiv_out : int*
//     Pivot indices.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgetrf_``.
LUCID_INTERNAL void
lapack_lu_factor_f64(const double* A, int n, double* LU_out, int* ipiv_out, int* info);

// Single-precision triangular linear solve.
//
// Solves $A X = B$ where $A$ is triangular.  Selects upper or lower triangle
// and whether the diagonal is implicit unit.  ``B`` is overwritten with $X$.
//
// Parameters
// ----------
// A : const float*
//     Row-major triangular matrix, $n \times n$.
// B : float*
//     Right-hand side, $n \times \text{nrhs}$ row-major; overwritten by $X$.
// n, nrhs : int
//     System dimensions.
// upper : bool
//     ``true`` if $A$ is upper triangular, else lower.
// unit : bool
//     ``true`` if the diagonal is implicit one (e.g., from packed LU).
// info : int*
//     LAPACK status (``> 0`` = exact-zero on diagonal).
//
// Math
// ----
// $$ A X = B \;\;\text{with $A$ triangular} $$
//
// References
// ----------
// LAPACK ``strtrs_``.
LUCID_INTERNAL void lapack_solve_triangular_f32(
    const float* A, float* B, int n, int nrhs, bool upper, bool unit, int* info);

// Double-precision triangular linear solve.  Uses ``dtrtrs_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major triangular matrix.
// B : double*
//     Right-hand side, overwritten by $X$.
// n, nrhs : int
//     System dimensions.
// upper, unit : bool
//     Triangle and unit-diagonal flags.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dtrtrs_``.
LUCID_INTERNAL void lapack_solve_triangular_f64(
    const double* A, double* B, int n, int nrhs, bool upper, bool unit, int* info);

// Single-precision least-squares solver via QR.
//
// Computes the minimum-2-norm solution of an overdetermined system:
// $$ \min_X \|A X - B\|_2 $$
// For tall $A$ (m ≥ n) uses QR; for wide $A$ (m < n) uses LQ.  ``B`` is
// overwritten with $X$ in its leading $n$ rows.
//
// Parameters
// ----------
// A : const float*
//     Row-major $m \times n$ input.
// B : float*
//     Row-major $m \times \text{nrhs}$ right-hand side; overwritten by $X$.
// m, n, nrhs : int
//     System dimensions.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``sgels_``.
LUCID_INTERNAL void lapack_lstsq_f32(const float* A, float* B, int m, int n, int nrhs, int* info);

// Double-precision least-squares solver.  Uses ``dgels_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major $m \times n$ input.
// B : double*
//     Row-major right-hand side; overwritten by $X$.
// m, n, nrhs : int
//     System dimensions.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgels_``.
LUCID_INTERNAL void lapack_lstsq_f64(const double* A, double* B, int m, int n, int nrhs, int* info);

// Single-precision LU-based linear solve with pre-computed factors.
//
// Given the packed LU factorisation produced by ``lapack_lu_factor_f32`` and
// its pivot indices, solves $A X = B$ for many right-hand sides without
// re-factoring.  ``B`` is overwritten with $X$.
//
// Parameters
// ----------
// LU : const float*
//     Packed LU matrix from ``lapack_lu_factor_f32``.
// ipiv : const int*
//     1-based pivot indices.
// B : float*
//     Row-major $n \times \text{nrhs}$; overwritten by $X$.
// n, nrhs : int
//     System dimensions.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``sgetrs_``.
LUCID_INTERNAL void
lapack_lu_solve_f32(const float* LU, const int* ipiv, float* B, int n, int nrhs, int* info);

// Double-precision LU-based linear solve with pre-computed factors.
// Uses ``dgetrs_``.
//
// Parameters
// ----------
// LU : const double*
//     Packed LU from ``lapack_lu_factor_f64``.
// ipiv : const int*
//     Pivot indices.
// B : double*
//     Right-hand side; overwritten by $X$.
// n, nrhs : int
//     System dimensions.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dgetrs_``.
LUCID_INTERNAL void
lapack_lu_solve_f64(const double* LU, const int* ipiv, double* B, int n, int nrhs, int* info);

// Single-precision Householder-product Q reconstruction.
//
// Given the Householder-reflector form of $Q$ produced by ``sgeqrf_``
// (matrix $H$ + scalar coefficients $\tau$), materialises the first $k$
// columns of $Q$ explicitly into ``Q_out``.
//
// Parameters
// ----------
// H : const float*
//     Row-major $m \times n$ reflector storage from ``sgeqrf_``.
// tau : const float*
//     Length-$k$ Householder scaling coefficients.
// Q_out : float*
//     Output orthonormal matrix, shape $m \times k$ (row-major).
// m, n, k : int
//     Dimensions; usually $k = \min(m, n)$.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``sorgqr_``.
LUCID_INTERNAL void lapack_householder_product_f32(
    const float* H, const float* tau, float* Q_out, int m, int n, int k, int* info);

// Double-precision Householder-product Q reconstruction.  Uses ``dorgqr_``.
//
// Parameters
// ----------
// H : const double*
//     Row-major reflector storage.
// tau : const double*
//     Householder scaling coefficients.
// Q_out : double*
//     Output $m \times k$ matrix.
// m, n, k : int
//     Dimensions.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dorgqr_``.
LUCID_INTERNAL void lapack_householder_product_f64(
    const double* H, const double* tau, double* Q_out, int m, int n, int k, int* info);

// Single-precision symmetric indefinite LDL^T factorisation.
//
// Bunch-Kaufman factorisation $A = L D L^T$ for a symmetric (possibly
// indefinite) matrix.  The packed $L$ + $D$ output is written to ``A_out``
// alongside pivot indices in ``ipiv``.
//
// Parameters
// ----------
// A : const float*
//     Row-major $n \times n$ symmetric input.
// A_out : float*
//     Packed factorisation (LAPACK ``ssytrf_`` native layout, row-major).
// ipiv : int*
//     Bunch-Kaufman pivot indices, length $n$.
// n : int
//     Side length.
// info : int*
//     LAPACK status word.
//
// Math
// ----
// $$ A = L \, D \, L^T $$
//
// References
// ----------
// LAPACK ``ssytrf_``.
LUCID_INTERNAL void
lapack_ldl_factor_f32(const float* A, float* A_out, int* ipiv, int n, int* info);

// Double-precision symmetric indefinite LDL^T factorisation.  Uses ``dsytrf_``.
//
// Parameters
// ----------
// A : const double*
//     Row-major symmetric input.
// A_out : double*
//     Packed LDL^T output.
// ipiv : int*
//     Bunch-Kaufman pivots.
// n : int
//     Side length.
// info : int*
//     LAPACK status word.
//
// References
// ----------
// LAPACK ``dsytrf_``.
LUCID_INTERNAL void
lapack_ldl_factor_f64(const double* A, double* A_out, int* ipiv, int n, int* info);

// Single-precision row-major → column-major transpose (LAPACK ingress).
//
// Used internally by the LAPACK wrappers to convert Lucid's row-major
// tensors into the column-major layout LAPACK expects.  Backed by
// ``vDSP_mtrans`` for SIMD-accelerated in-cache transposition.
//
// Parameters
// ----------
// src : const float*
//     Source row-major matrix, ``rows × cols``.
// dst : float*
//     Destination column-major buffer of identical element count.
// rows, cols : int
//     Source dimensions (row count and column count of the row-major input).
//
// References
// ----------
// Accelerate.framework ``vDSP_mtrans``.
LUCID_INTERNAL void transpose_to_col_major_f32(const float* src, float* dst, int rows, int cols);

// Double-precision row-major → column-major transpose.  Uses ``vDSP_mtransD``.
//
// Parameters
// ----------
// src : const double*
//     Source row-major matrix.
// dst : double*
//     Destination column-major buffer.
// rows, cols : int
//     Source dimensions.
//
// References
// ----------
// Accelerate.framework ``vDSP_mtransD``.
LUCID_INTERNAL void transpose_to_col_major_f64(const double* src, double* dst, int rows, int cols);

// Single-precision column-major → row-major transpose (LAPACK egress).
//
// Inverse of ``transpose_to_col_major_f32``; turns a LAPACK column-major
// output back into a Lucid row-major tensor.
//
// Parameters
// ----------
// src : const float*
//     LAPACK column-major buffer.
// dst : float*
//     Destination row-major buffer.
// rows, cols : int
//     Row-major output dimensions.
//
// References
// ----------
// Accelerate.framework ``vDSP_mtrans``.
LUCID_INTERNAL void transpose_from_col_major_f32(const float* src, float* dst, int rows, int cols);

// Double-precision column-major → row-major transpose.  Uses ``vDSP_mtransD``.
//
// Parameters
// ----------
// src : const double*
//     LAPACK column-major buffer.
// dst : double*
//     Destination row-major buffer.
// rows, cols : int
//     Row-major output dimensions.
//
// References
// ----------
// Accelerate.framework ``vDSP_mtransD``.
LUCID_INTERNAL void
transpose_from_col_major_f64(const double* src, double* dst, int rows, int cols);

}  // namespace lucid::backend::cpu
