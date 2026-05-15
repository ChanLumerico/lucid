// lucid/_C/backend/cpu/Lapack.cpp
//
// Implements the LAPACK wrappers declared in Lapack.h.  Each function:
//   1. Copies the input to a temporary column-major buffer using vDSP_mtrans.
//   2. Calls the Accelerate LAPACK routine with lwork=-1 to query workspace
//      size, allocates the workspace, then calls the routine again.
//   3. Converts results back to row-major and writes them to the output buffers.
//
// The column-major round-trip is unavoidable because LAPACK was written in
// Fortran, which uses column-major ("Fortran order") storage natively.  Using
// vDSP_mtrans for the conversion avoids scalar loops in the hot path.

#include "Lapack.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <Accelerate/Accelerate.h>

namespace lucid::backend::cpu {

namespace {

// Accelerate's new LAPACK interface (-DACCELERATE_NEW_LAPACK) exposes
// __LAPACK_int as the INTEGER argument type — int in LP64 mode, long in
// ILP64. The legacy __CLPK_integer alias is gone in the new headers.
using i32 = __LAPACK_int;

// Transposes a rows×cols row-major matrix to cols×rows column-major layout.
// vDSP_mtrans(src, 1, dst, 1, cols, rows) reads rows rows of length cols
// and writes them as columns, which is equivalent to a full matrix transpose.
inline void rowmajor_to_colmajor_f32(const float* src, float* dst, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, cols, rows);
}

inline void rowmajor_to_colmajor_f64(const double* src, double* dst, int rows, int cols) {
    vDSP_mtransD(src, 1, dst, 1, cols, rows);
}

// Transposes a cols×rows column-major matrix back to rows×cols row-major layout.
inline void colmajor_to_rowmajor_f32(const float* src, float* dst, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, rows, cols);
}

inline void colmajor_to_rowmajor_f64(const double* src, double* dst, int rows, int cols) {
    vDSP_mtransD(src, 1, dst, 1, rows, cols);
}

}  // namespace

void transpose_to_col_major_f32(const float* src, float* dst, int rows, int cols) {
    rowmajor_to_colmajor_f32(src, dst, rows, cols);
}

void transpose_to_col_major_f64(const double* src, double* dst, int rows, int cols) {
    rowmajor_to_colmajor_f64(src, dst, rows, cols);
}

void transpose_from_col_major_f32(const float* src, float* dst, int rows, int cols) {
    colmajor_to_rowmajor_f32(src, dst, rows, cols);
}

void transpose_from_col_major_f64(const double* src, double* dst, int rows, int cols) {
    colmajor_to_rowmajor_f64(src, dst, rows, cols);
}

void lapack_inv_f32(float* A, int n, int* info_out) {
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);

    std::vector<i32> ipiv(n);
    i32 N = n;
    i32 lda = n;
    i32 info = 0;
    sgetrf_(&N, &N, Ac.data(), &lda, ipiv.data(), &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    i32 lwork = -1;
    float wkopt;
    sgetri_(&N, Ac.data(), &lda, ipiv.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    if (lwork < n)
        lwork = n;
    std::vector<float> work(static_cast<std::size_t>(lwork));
    sgetri_(&N, Ac.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f32(Ac.data(), A, n, n);
    *info_out = 0;
}

void lapack_inv_f64(double* A, int n, int* info_out) {
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);

    std::vector<i32> ipiv(n);
    i32 N = n;
    i32 lda = n;
    i32 info = 0;
    dgetrf_(&N, &N, Ac.data(), &lda, ipiv.data(), &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    i32 lwork = -1;
    double wkopt;
    dgetri_(&N, Ac.data(), &lda, ipiv.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    if (lwork < n)
        lwork = n;
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgetri_(&N, Ac.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f64(Ac.data(), A, n, n);
    *info_out = 0;
}

void lapack_solve_f32(float* A, float* B, int n, int nrhs, int* info_out) {
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    std::vector<float> Bc(static_cast<std::size_t>(n) * nrhs);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);
    rowmajor_to_colmajor_f32(B, Bc.data(), n, nrhs);

    std::vector<i32> ipiv(n);
    i32 N = n, NRHS = nrhs, lda = n, ldb = n;
    i32 info = 0;
    sgesv_(&N, &NRHS, Ac.data(), &lda, ipiv.data(), Bc.data(), &ldb, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f32(Bc.data(), B, n, nrhs);
    *info_out = 0;
}

void lapack_solve_f64(double* A, double* B, int n, int nrhs, int* info_out) {
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    std::vector<double> Bc(static_cast<std::size_t>(n) * nrhs);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);
    rowmajor_to_colmajor_f64(B, Bc.data(), n, nrhs);

    std::vector<i32> ipiv(n);
    i32 N = n, NRHS = nrhs, lda = n, ldb = n;
    i32 info = 0;
    dgesv_(&N, &NRHS, Ac.data(), &lda, ipiv.data(), Bc.data(), &ldb, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f64(Bc.data(), B, n, nrhs);
    *info_out = 0;
}

namespace {

template <typename T>
void split_lu(const T* LU_col, int n, T* L_out, T* U_out) {
    std::vector<T> LU_row(static_cast<std::size_t>(n) * n);
    if constexpr (std::is_same_v<T, float>)
        colmajor_to_rowmajor_f32(LU_col, LU_row.data(), n, n);
    else
        colmajor_to_rowmajor_f64(LU_col, LU_row.data(), n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const T v = LU_row[i * n + j];
            if (i > j) {
                L_out[i * n + j] = v;
                U_out[i * n + j] = T{0};
            } else if (i < j) {
                L_out[i * n + j] = T{0};
                U_out[i * n + j] = v;
            } else {
                L_out[i * n + j] = T{1};
                U_out[i * n + j] = v;
            }
        }
    }
}

}  // namespace

void lapack_lu_f32(const float* A, int n, int* ipiv, float* L_out, float* U_out, int* info_out) {
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);

    std::vector<i32> ipiv_local(n);
    i32 N = n, lda = n, info = 0;
    sgetrf_(&N, &N, Ac.data(), &lda, ipiv_local.data(), &info);
    if (info < 0) {
        *info_out = info;
        return;
    }

    for (int i = 0; i < n; ++i)
        ipiv[i] = ipiv_local[i];
    split_lu(Ac.data(), n, L_out, U_out);
    *info_out = info;
}

void lapack_lu_f64(const double* A, int n, int* ipiv, double* L_out, double* U_out, int* info_out) {
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);

    std::vector<i32> ipiv_local(n);
    i32 N = n, lda = n, info = 0;
    dgetrf_(&N, &N, Ac.data(), &lda, ipiv_local.data(), &info);
    if (info < 0) {
        *info_out = info;
        return;
    }

    for (int i = 0; i < n; ++i)
        ipiv[i] = ipiv_local[i];
    split_lu(Ac.data(), n, L_out, U_out);
    *info_out = info;
}

void lapack_cholesky_f32(float* A, int n, bool lower, int* info_out) {
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);

    char uplo = lower ? 'L' : 'U';
    i32 N = n, lda = n, info = 0;
    spotrf_(&uplo, &N, Ac.data(), &lda, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f32(Ac.data(), A, n, n);

    if (lower) {
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                A[i * n + j] = 0.0f;
    } else {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j)
                A[i * n + j] = 0.0f;
    }
    *info_out = 0;
}

void lapack_cholesky_f64(double* A, int n, bool lower, int* info_out) {
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);

    char uplo = lower ? 'L' : 'U';
    i32 N = n, lda = n, info = 0;
    dpotrf_(&uplo, &N, Ac.data(), &lda, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f64(Ac.data(), A, n, n);
    if (lower) {
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                A[i * n + j] = 0.0;
    } else {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j)
                A[i * n + j] = 0.0;
    }
    *info_out = 0;
}

void lapack_qr_f32(const float* A, int m, int n, float* Q, float* R, int* info_out) {
    const int k = std::min(m, n);
    std::vector<float> Ac(static_cast<std::size_t>(m) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), m, n);

    std::vector<float> tau(static_cast<std::size_t>(k));
    i32 M = m, N = n, lda = m, info = 0;

    i32 lwork = -1;
    float wkopt;
    sgeqrf_(&M, &N, Ac.data(), &lda, tau.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    if (lwork < std::max(1, n))
        lwork = std::max(1, n);
    std::vector<float> work(static_cast<std::size_t>(lwork));
    sgeqrf_(&M, &N, Ac.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    std::vector<float> R_row(static_cast<std::size_t>(k) * n, 0.0f);
    for (int i = 0; i < k; ++i)
        for (int j = i; j < n; ++j)
            R_row[i * n + j] = Ac[j * m + i];
    std::memcpy(R, R_row.data(), R_row.size() * sizeof(float));

    i32 K = k;
    lwork = -1;
    sorgqr_(&M, &K, &K, Ac.data(), &lda, tau.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    if (lwork < std::max(1, k))
        lwork = std::max(1, k);
    work.resize(static_cast<std::size_t>(lwork));
    sorgqr_(&M, &K, &K, Ac.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            Q[i * k + j] = Ac[j * m + i];

    *info_out = 0;
}

void lapack_qr_f64(const double* A, int m, int n, double* Q, double* R, int* info_out) {
    const int k = std::min(m, n);
    std::vector<double> Ac(static_cast<std::size_t>(m) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), m, n);

    std::vector<double> tau(static_cast<std::size_t>(k));
    i32 M = m, N = n, lda = m, info = 0;

    i32 lwork = -1;
    double wkopt;
    dgeqrf_(&M, &N, Ac.data(), &lda, tau.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    if (lwork < std::max(1, n))
        lwork = std::max(1, n);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeqrf_(&M, &N, Ac.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    std::vector<double> R_row(static_cast<std::size_t>(k) * n, 0.0);
    for (int i = 0; i < k; ++i)
        for (int j = i; j < n; ++j)
            R_row[i * n + j] = Ac[j * m + i];
    std::memcpy(R, R_row.data(), R_row.size() * sizeof(double));

    i32 K = k;
    lwork = -1;
    dorgqr_(&M, &K, &K, Ac.data(), &lda, tau.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    if (lwork < std::max(1, k))
        lwork = std::max(1, k);
    work.resize(static_cast<std::size_t>(lwork));
    dorgqr_(&M, &K, &K, Ac.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            Q[i * k + j] = Ac[j * m + i];

    *info_out = 0;
}

void lapack_svd_f32(const float* A,
                    int m,
                    int n,
                    bool full_matrices,
                    float* U_out,
                    float* S_out,
                    float* Vt_out,
                    int* info_out) {
    const int k = std::min(m, n);
    std::vector<float> Ac(static_cast<std::size_t>(m) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), m, n);

    char jobz = full_matrices ? 'A' : 'S';
    const int u_cols = full_matrices ? m : k;
    const int vt_rows = full_matrices ? n : k;
    std::vector<float> Uc(static_cast<std::size_t>(m) * u_cols);
    std::vector<float> Vtc(static_cast<std::size_t>(vt_rows) * n);
    std::vector<float> Sv(k);

    i32 M = m, N = n, lda = m, ldu = m, ldvt = vt_rows, info = 0;
    std::vector<i32> iwork(8 * k);

    i32 lwork = -1;
    float wkopt;
    sgesdd_(&jobz, &M, &N, Ac.data(), &lda, Sv.data(), Uc.data(), &ldu, Vtc.data(), &ldvt, &wkopt,
            &lwork, iwork.data(), &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<float> work(static_cast<std::size_t>(lwork));
    sgesdd_(&jobz, &M, &N, Ac.data(), &lda, Sv.data(), Uc.data(), &ldu, Vtc.data(), &ldvt,
            work.data(), &lwork, iwork.data(), &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f32(Uc.data(), U_out, m, u_cols);

    std::memcpy(S_out, Sv.data(), k * sizeof(float));

    colmajor_to_rowmajor_f32(Vtc.data(), Vt_out, vt_rows, n);

    *info_out = 0;
}

void lapack_svd_f64(const double* A,
                    int m,
                    int n,
                    bool full_matrices,
                    double* U_out,
                    double* S_out,
                    double* Vt_out,
                    int* info_out) {
    const int k = std::min(m, n);
    std::vector<double> Ac(static_cast<std::size_t>(m) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), m, n);

    char jobz = full_matrices ? 'A' : 'S';
    const int u_cols = full_matrices ? m : k;
    const int vt_rows = full_matrices ? n : k;
    std::vector<double> Uc(static_cast<std::size_t>(m) * u_cols);
    std::vector<double> Vtc(static_cast<std::size_t>(vt_rows) * n);
    std::vector<double> Sv(k);

    i32 M = m, N = n, lda = m, ldu = m, ldvt = vt_rows, info = 0;
    std::vector<i32> iwork(8 * k);

    i32 lwork = -1;
    double wkopt;
    dgesdd_(&jobz, &M, &N, Ac.data(), &lda, Sv.data(), Uc.data(), &ldu, Vtc.data(), &ldvt, &wkopt,
            &lwork, iwork.data(), &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgesdd_(&jobz, &M, &N, Ac.data(), &lda, Sv.data(), Uc.data(), &ldu, Vtc.data(), &ldvt,
            work.data(), &lwork, iwork.data(), &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f64(Uc.data(), U_out, m, u_cols);
    std::memcpy(S_out, Sv.data(), k * sizeof(double));
    colmajor_to_rowmajor_f64(Vtc.data(), Vt_out, vt_rows, n);

    *info_out = 0;
}

void lapack_eigh_f32(const float* A, int n, float* w, float* V_out, int* info_out) {
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);

    char jobz = 'V', uplo = 'L';
    i32 N = n, lda = n, info = 0;

    i32 lwork = -1, liwork = -1;
    float wkopt;
    i32 iwkopt;
    ssyevd_(&jobz, &uplo, &N, Ac.data(), &lda, w, &wkopt, &lwork, &iwkopt, &liwork, &info);
    lwork = static_cast<i32>(wkopt);
    liwork = iwkopt;
    std::vector<float> work(static_cast<std::size_t>(lwork));
    std::vector<i32> iwork(static_cast<std::size_t>(liwork));
    ssyevd_(&jobz, &uplo, &N, Ac.data(), &lda, w, work.data(), &lwork, iwork.data(), &liwork,
            &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            V_out[i * n + j] = Ac[j * n + i];

    *info_out = 0;
}

void lapack_eigh_f64(const double* A, int n, double* w, double* V_out, int* info_out) {
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);

    char jobz = 'V', uplo = 'L';
    i32 N = n, lda = n, info = 0;

    i32 lwork = -1, liwork = -1;
    double wkopt;
    i32 iwkopt;
    dsyevd_(&jobz, &uplo, &N, Ac.data(), &lda, w, &wkopt, &lwork, &iwkopt, &liwork, &info);
    lwork = static_cast<i32>(wkopt);
    liwork = iwkopt;
    std::vector<double> work(static_cast<std::size_t>(lwork));
    std::vector<i32> iwork(static_cast<std::size_t>(liwork));
    dsyevd_(&jobz, &uplo, &N, Ac.data(), &lda, w, work.data(), &lwork, iwork.data(), &liwork,
            &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            V_out[i * n + j] = Ac[j * n + i];

    *info_out = 0;
}

void lapack_eig_f32(const float* A, int n, float* wr, float* wi, float* VR, int* info_out) {
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);

    char jobvl = 'N', jobvr = (VR ? 'V' : 'N');
    i32 N = n, lda = n, ldvl = 1, ldvr = (VR ? n : 1), info = 0;

    std::vector<float> VL_dummy(1);
    std::vector<float> VRc(static_cast<std::size_t>(n) * (VR ? n : 0));

    i32 lwork = -1;
    float wkopt;
    sgeev_(&jobvl, &jobvr, &N, Ac.data(), &lda, wr, wi, VL_dummy.data(), &ldvl, VRc.data(), &ldvr,
           &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<float> work(static_cast<std::size_t>(lwork));
    sgeev_(&jobvl, &jobvr, &N, Ac.data(), &lda, wr, wi, VL_dummy.data(), &ldvl, VRc.data(), &ldvr,
           work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    if (VR) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                VR[i * n + j] = VRc[j * n + i];
    }
    *info_out = 0;
}

void lapack_eig_f64(const double* A, int n, double* wr, double* wi, double* VR, int* info_out) {
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);

    char jobvl = 'N', jobvr = (VR ? 'V' : 'N');
    i32 N = n, lda = n, ldvl = 1, ldvr = (VR ? n : 1), info = 0;

    std::vector<double> VL_dummy(1);
    std::vector<double> VRc(static_cast<std::size_t>(n) * (VR ? n : 0));

    i32 lwork = -1;
    double wkopt;
    dgeev_(&jobvl, &jobvr, &N, Ac.data(), &lda, wr, wi, VL_dummy.data(), &ldvl, VRc.data(), &ldvr,
           &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeev_(&jobvl, &jobvr, &N, Ac.data(), &lda, wr, wi, VL_dummy.data(), &ldvl, VRc.data(), &ldvr,
           work.data(), &lwork, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    if (VR) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                VR[i * n + j] = VRc[j * n + i];
    }
    *info_out = 0;
}

// LU factorisation (packed format): returns the raw dgetrf_ output where the
// unit-lower and upper triangular factors share a single matrix (LAPACK packed
// format), plus the 1-based pivot index array.
//
// Out layout matches reference framework's lu_factor / the reference LU factor API:
//   LU_out  : n×n row-major packed LU  (L below diagonal, U on/above diagonal,
//             implicit unit diagonal of L)
//   ipiv_out: n int32_t pivot indices (1-based, matching LAPACK convention)
void lapack_lu_factor_f32(const float* A, int n, float* LU_out, int* ipiv_out, int* info_out) {
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);

    std::vector<i32> ipiv_local(n);
    i32 N = n, lda = n, info = 0;
    sgetrf_(&N, &N, Ac.data(), &lda, ipiv_local.data(), &info);
    if (info < 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f32(Ac.data(), LU_out, n, n);
    for (int i = 0; i < n; ++i)
        ipiv_out[i] = static_cast<int>(ipiv_local[i]);
    *info_out = info;
}

void lapack_lu_factor_f64(const double* A, int n, double* LU_out, int* ipiv_out, int* info_out) {
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);

    std::vector<i32> ipiv_local(n);
    i32 N = n, lda = n, info = 0;
    dgetrf_(&N, &N, Ac.data(), &lda, ipiv_local.data(), &info);
    if (info < 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f64(Ac.data(), LU_out, n, n);
    for (int i = 0; i < n; ++i)
        ipiv_out[i] = static_cast<int>(ipiv_local[i]);
    *info_out = info;
}

// Triangular solve: solve A X = B (or Aᵀ X = B) where A is triangular.
// Overwrites B with the solution X.
// upper=true  → A is upper triangular; upper=false → lower triangular.
// unit=true   → diagonal of A is treated as all-ones (unit triangular).
// Uses LAPACK strtrs_ / dtrtrs_.
void lapack_solve_triangular_f32(
    const float* A, float* B, int n, int nrhs, bool upper, bool unit, int* info_out) {
    char uplo = upper ? 'U' : 'L';
    char diag = unit ? 'U' : 'N';
    char trans = 'N';

    // LAPACK expects column-major input.
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    std::vector<float> Bc(static_cast<std::size_t>(n) * nrhs);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);
    rowmajor_to_colmajor_f32(B, Bc.data(), n, nrhs);

    i32 N = n, NRHS = nrhs, lda = n, ldb = n, info = 0;
    strtrs_(&uplo, &trans, &diag, &N, &NRHS, Ac.data(), &lda, Bc.data(), &ldb, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f32(Bc.data(), B, n, nrhs);
    *info_out = 0;
}

void lapack_solve_triangular_f64(
    const double* A, double* B, int n, int nrhs, bool upper, bool unit, int* info_out) {
    char uplo = upper ? 'U' : 'L';
    char diag = unit ? 'U' : 'N';
    char trans = 'N';

    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    std::vector<double> Bc(static_cast<std::size_t>(n) * nrhs);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);
    rowmajor_to_colmajor_f64(B, Bc.data(), n, nrhs);

    i32 N = n, NRHS = nrhs, lda = n, ldb = n, info = 0;
    dtrtrs_(&uplo, &trans, &diag, &N, &NRHS, Ac.data(), &lda, Bc.data(), &ldb, &info);
    if (info != 0) {
        *info_out = info;
        return;
    }

    colmajor_to_rowmajor_f64(Bc.data(), B, n, nrhs);
    *info_out = 0;
}

// ── lstsq ────────────────────────────────────────────────────────────────────
// Uses sgels_/dgels_ to solve min||AX-B||_2 (overdetermined or underdetermined).
// A is m×n; B is m×nrhs on entry, solution X in first n rows on exit.
// Both A and B are modified in-place.

void lapack_lstsq_f32(const float* A, float* B, int m, int n, int nrhs, int* info_out) {
    char trans = 'N';
    std::vector<float> Ac(static_cast<std::size_t>(m) * n);
    std::vector<float> Bc(static_cast<std::size_t>(std::max(m, n)) * nrhs, 0.0f);
    rowmajor_to_colmajor_f32(A, Ac.data(), m, n);
    rowmajor_to_colmajor_f32(B, Bc.data(), m, nrhs);

    i32 M = m, N = n, NRHS = nrhs, lda = m, ldb = std::max(m, n), info = 0;
    float wkopt = 0.0f;
    i32 lwork = -1;
    sgels_(&trans, &M, &N, &NRHS, Ac.data(), &lda, Bc.data(), &ldb, &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<float> work(static_cast<std::size_t>(lwork));
    sgels_(&trans, &M, &N, &NRHS, Ac.data(), &lda, Bc.data(), &ldb, work.data(), &lwork, &info);
    *info_out = info;
    if (info == 0)
        colmajor_to_rowmajor_f32(Bc.data(), B, std::max(m, n), nrhs);
}

void lapack_lstsq_f64(const double* A, double* B, int m, int n, int nrhs, int* info_out) {
    char trans = 'N';
    std::vector<double> Ac(static_cast<std::size_t>(m) * n);
    std::vector<double> Bc(static_cast<std::size_t>(std::max(m, n)) * nrhs, 0.0);
    rowmajor_to_colmajor_f64(A, Ac.data(), m, n);
    rowmajor_to_colmajor_f64(B, Bc.data(), m, nrhs);

    i32 M = m, N = n, NRHS = nrhs, lda = m, ldb = std::max(m, n), info = 0;
    double wkopt = 0.0;
    i32 lwork = -1;
    dgels_(&trans, &M, &N, &NRHS, Ac.data(), &lda, Bc.data(), &ldb, &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgels_(&trans, &M, &N, &NRHS, Ac.data(), &lda, Bc.data(), &ldb, work.data(), &lwork, &info);
    *info_out = info;
    if (info == 0)
        colmajor_to_rowmajor_f64(Bc.data(), B, std::max(m, n), nrhs);
}

// ── lu_solve ──────────────────────────────────────────────────────────────────
// Solve AX=B given LU packed matrix and 1-based pivot vector from lu_factor.
// Uses sgetrs_/dgetrs_. B is overwritten with X.

void lapack_lu_solve_f32(
    const float* LU, const int* ipiv, float* B, int n, int nrhs, int* info_out) {
    char trans = 'N';
    std::vector<float> LUc(static_cast<std::size_t>(n) * n);
    std::vector<float> Bc(static_cast<std::size_t>(n) * nrhs);
    rowmajor_to_colmajor_f32(LU, LUc.data(), n, n);
    rowmajor_to_colmajor_f32(B, Bc.data(), n, nrhs);

    // Convert 1-based int32 ipiv to LAPACK i32 (also 1-based)
    std::vector<i32> piv(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        piv[static_cast<std::size_t>(i)] = static_cast<i32>(ipiv[i]);

    i32 N = n, NRHS = nrhs, lda = n, ldb = n, info = 0;
    sgetrs_(&trans, &N, &NRHS, LUc.data(), &lda, piv.data(), Bc.data(), &ldb, &info);
    *info_out = info;
    if (info == 0)
        colmajor_to_rowmajor_f32(Bc.data(), B, n, nrhs);
}

void lapack_lu_solve_f64(
    const double* LU, const int* ipiv, double* B, int n, int nrhs, int* info_out) {
    char trans = 'N';
    std::vector<double> LUc(static_cast<std::size_t>(n) * n);
    std::vector<double> Bc(static_cast<std::size_t>(n) * nrhs);
    rowmajor_to_colmajor_f64(LU, LUc.data(), n, n);
    rowmajor_to_colmajor_f64(B, Bc.data(), n, nrhs);

    std::vector<i32> piv(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        piv[static_cast<std::size_t>(i)] = static_cast<i32>(ipiv[i]);

    i32 N = n, NRHS = nrhs, lda = n, ldb = n, info = 0;
    dgetrs_(&trans, &N, &NRHS, LUc.data(), &lda, piv.data(), Bc.data(), &ldb, &info);
    *info_out = info;
    if (info == 0)
        colmajor_to_rowmajor_f64(Bc.data(), B, n, nrhs);
}

// ── householder_product ───────────────────────────────────────────────────────
// Reconstruct Q (m×k) from Householder reflectors stored in H and tau.
// H is m×n (geqrf output), tau is length k=min(m,n). Uses sorgqr_/dorgqr_.

void lapack_householder_product_f32(
    const float* H, const float* tau, float* Q_out, int m, int n, int k, int* info_out) {
    std::vector<float> Hc(static_cast<std::size_t>(m) * n);
    rowmajor_to_colmajor_f32(H, Hc.data(), m, n);

    i32 M = m, N = k, K = k, lda = m, info = 0;
    float wkopt = 0.0f;
    i32 lwork = -1;
    sorgqr_(&M, &N, &K, Hc.data(), &lda, const_cast<float*>(tau), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<float> work(static_cast<std::size_t>(lwork));
    sorgqr_(&M, &N, &K, Hc.data(), &lda, const_cast<float*>(tau), work.data(), &lwork, &info);
    *info_out = info;
    if (info == 0)
        colmajor_to_rowmajor_f32(Hc.data(), Q_out, m, k);
}

void lapack_householder_product_f64(
    const double* H, const double* tau, double* Q_out, int m, int n, int k, int* info_out) {
    std::vector<double> Hc(static_cast<std::size_t>(m) * n);
    rowmajor_to_colmajor_f64(H, Hc.data(), m, n);

    i32 M = m, N = k, K = k, lda = m, info = 0;
    double wkopt = 0.0;
    i32 lwork = -1;
    dorgqr_(&M, &N, &K, Hc.data(), &lda, const_cast<double*>(tau), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dorgqr_(&M, &N, &K, Hc.data(), &lda, const_cast<double*>(tau), work.data(), &lwork, &info);
    *info_out = info;
    if (info == 0)
        colmajor_to_rowmajor_f64(Hc.data(), Q_out, m, k);
}

// ── ldl_factor ────────────────────────────────────────────────────────────────
// LDL^T factorization of symmetric n×n matrix A. Uses ssytrf_/dsytrf_.
// A_out receives the packed LAPACK result; ipiv receives 1-based pivot indices.

void lapack_ldl_factor_f32(const float* A, float* A_out, int* ipiv, int n, int* info_out) {
    char uplo = 'L';
    std::vector<float> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f32(A, Ac.data(), n, n);

    std::vector<i32> piv(static_cast<std::size_t>(n));
    float wkopt = 0.0f;
    i32 N = n, lda = n, info = 0, lwork = -1;
    ssytrf_(&uplo, &N, Ac.data(), &lda, piv.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<float> work(static_cast<std::size_t>(lwork));
    ssytrf_(&uplo, &N, Ac.data(), &lda, piv.data(), work.data(), &lwork, &info);
    *info_out = info;
    if (info == 0) {
        colmajor_to_rowmajor_f32(Ac.data(), A_out, n, n);
        for (int i = 0; i < n; ++i)
            ipiv[i] = static_cast<int>(piv[static_cast<std::size_t>(i)]);
    }
}

void lapack_ldl_factor_f64(const double* A, double* A_out, int* ipiv, int n, int* info_out) {
    char uplo = 'L';
    std::vector<double> Ac(static_cast<std::size_t>(n) * n);
    rowmajor_to_colmajor_f64(A, Ac.data(), n, n);

    std::vector<i32> piv(static_cast<std::size_t>(n));
    double wkopt = 0.0;
    i32 N = n, lda = n, info = 0, lwork = -1;
    dsytrf_(&uplo, &N, Ac.data(), &lda, piv.data(), &wkopt, &lwork, &info);
    lwork = static_cast<i32>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsytrf_(&uplo, &N, Ac.data(), &lda, piv.data(), work.data(), &lwork, &info);
    *info_out = info;
    if (info == 0) {
        colmajor_to_rowmajor_f64(Ac.data(), A_out, n, n);
        for (int i = 0; i < n; ++i)
            ipiv[i] = static_cast<int>(piv[static_cast<std::size_t>(i)]);
    }
}

}  // namespace lucid::backend::cpu
