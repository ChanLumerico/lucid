#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

LUCID_INTERNAL void lapack_inv_f32(float* A, int n, int* info);
LUCID_INTERNAL void lapack_inv_f64(double* A, int n, int* info);

LUCID_INTERNAL void lapack_solve_f32(float* A, float* B, int n, int nrhs, int* info);
LUCID_INTERNAL void lapack_solve_f64(double* A, double* B, int n, int nrhs, int* info);

LUCID_INTERNAL void
lapack_lu_f32(const float* A, int n, int* ipiv, float* L_out, float* U_out, int* info);
LUCID_INTERNAL void
lapack_lu_f64(const double* A, int n, int* ipiv, double* L_out, double* U_out, int* info);

LUCID_INTERNAL void lapack_cholesky_f32(float* A, int n, bool lower, int* info);
LUCID_INTERNAL void lapack_cholesky_f64(double* A, int n, bool lower, int* info);

LUCID_INTERNAL void lapack_qr_f32(const float* A, int m, int n, float* Q, float* R, int* info);
LUCID_INTERNAL void lapack_qr_f64(const double* A, int m, int n, double* Q, double* R, int* info);

LUCID_INTERNAL void lapack_svd_f32(
    const float* A, int m, int n, bool full_matrices, float* U, float* S, float* Vt, int* info);
LUCID_INTERNAL void lapack_svd_f64(
    const double* A, int m, int n, bool full_matrices, double* U, double* S, double* Vt, int* info);

LUCID_INTERNAL void lapack_eigh_f32(const float* A, int n, float* w, float* V_out, int* info);
LUCID_INTERNAL void lapack_eigh_f64(const double* A, int n, double* w, double* V_out, int* info);

LUCID_INTERNAL void
lapack_eig_f32(const float* A, int n, float* wr, float* wi, float* VR, int* info);
LUCID_INTERNAL void
lapack_eig_f64(const double* A, int n, double* wr, double* wi, double* VR, int* info);

LUCID_INTERNAL void transpose_to_col_major_f32(const float* src, float* dst, int rows, int cols);
LUCID_INTERNAL void transpose_to_col_major_f64(const double* src, double* dst, int rows, int cols);
LUCID_INTERNAL void transpose_from_col_major_f32(const float* src, float* dst, int rows, int cols);
LUCID_INTERNAL void
transpose_from_col_major_f64(const double* src, double* dst, int rows, int cols);

}  // namespace lucid::backend::cpu
