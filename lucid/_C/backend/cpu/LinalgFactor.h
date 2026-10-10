// lucid/_C/backend/cpu/LinalgFactor.h
//
// The pivoted factorisations that complete on a singular matrix — ``getrf``
// (LU) and ``sytrf`` (Bunch-Kaufman LDLᵀ) — and the ``info`` they report.
//
// ``info > 0`` from either is not a failure to produce an answer: ``getrf``
// has met an exactly-zero pivot of U, ``sytrf`` a zero (or NaN) block of D,
// and both have finished the factorisation.  Whether that is an error is the
// caller's decision (``lu_factor`` raises, ``lu_factor_ex`` reports), so the
// kernel returns the factor, the pivots and LAPACK's own ``info`` for every
// matrix of the batch.  Reading ``info`` back off the factor instead was
// wrong for non-finite input, where LAPACK's tests on NaN and infinity are
// not a property of the finished factor (LCD-244 review).  Only ``info < 0``
// — an argument Lucid marshalled wrongly — raises here.  The Metal stream
// delegates to these, so both devices report the same ``info``.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../LapackInfo.h"
#include "Lapack.h"

namespace lucid::backend::cpu {

static_assert(sizeof(int) == sizeof(std::int32_t), "LAPACK pivots are written as int32");

// ``factor_one(in, out, ipiv, &info)`` over every matrix of ``a``: returns
// ``{factor, pivots, info}`` — the factor shaped like ``a``, ``pivots``
// ``(batch, piv_len)`` int32, ``info`` one int32 per matrix.
template <typename T, typename FactorOne>
std::vector<Storage> pivoted_factor(const CpuStorage& a,
                                    const Shape& shape,
                                    int piv_len,
                                    Dtype dt,
                                    const char* op,
                                    FactorOne&& factor_one) {
    std::int64_t batch = 1;
    for (std::size_t i = 0; i + 2 < shape.size(); ++i)
        batch *= shape[i];
    const std::size_t per_mat =
        static_cast<std::size_t>(shape[shape.size() - 2] * shape[shape.size() - 1]);
    const std::size_t piv_nb = static_cast<std::size_t>(batch) * piv_len * sizeof(std::int32_t);
    const std::size_t info_nb = static_cast<std::size_t>(batch) * sizeof(std::int32_t);
    auto out = allocate_aligned_bytes(a.nbytes, Device::CPU);
    auto piv = allocate_aligned_bytes(piv_nb, Device::CPU);
    auto info = allocate_aligned_bytes(info_nb, Device::CPU);
    const auto* src = reinterpret_cast<const T*>(a.ptr.get());
    auto* dst = reinterpret_cast<T*>(out.get());
    auto* piv_out = reinterpret_cast<int*>(piv.get());
    auto* info_out = reinterpret_cast<std::int32_t*>(info.get());
    for (std::int64_t b = 0; b < batch; ++b) {
        int status = 0;
        factor_one(src + b * per_mat, dst + b * per_mat, piv_out + b * piv_len, &status);
        check_lapack_info(status < 0 ? status : 0, op, b, batch);
        info_out[b] = status;
    }
    return {Storage{CpuStorage{out, a.nbytes, dt}}, Storage{CpuStorage{piv, piv_nb, Dtype::I32}},
            Storage{CpuStorage{info, info_nb, Dtype::I32}}};
}

// ``getrf`` of every (m, n) matrix: ``{LU, pivots (batch, min(m, n)), info}``.
inline std::vector<Storage> lu_factor(const CpuStorage& a, const Shape& shape, Dtype dt) {
    const int m = static_cast<int>(shape[shape.size() - 2]);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const int k = m < n ? m : n;
    if (dt == Dtype::F32)
        return pivoted_factor<float>(a, shape, k, dt, "lu_factor",
                                     [&](const float* in, float* out, int* ipiv, int* info) {
                                         lapack_lu_factor_f32(in, m, n, out, ipiv, info);
                                     });
    return pivoted_factor<double>(a, shape, k, dt, "lu_factor",
                                  [&](const double* in, double* out, int* ipiv, int* info) {
                                      lapack_lu_factor_f64(in, m, n, out, ipiv, info);
                                  });
}

// ``sytrf`` of every (n, n) matrix: ``{LD, pivots (batch, n), info}``.
inline std::vector<Storage> ldl_factor(const CpuStorage& a, const Shape& shape, Dtype dt) {
    const int n = static_cast<int>(shape[shape.size() - 1]);
    if (dt == Dtype::F32)
        return pivoted_factor<float>(a, shape, n, dt, "ldl_factor",
                                     [&](const float* in, float* out, int* ipiv, int* info) {
                                         lapack_ldl_factor_f32(in, out, ipiv, n, info);
                                     });
    return pivoted_factor<double>(a, shape, n, dt, "ldl_factor",
                                  [&](const double* in, double* out, int* ipiv, int* info) {
                                      lapack_ldl_factor_f64(in, out, ipiv, n, info);
                                  });
}

}  // namespace lucid::backend::cpu
