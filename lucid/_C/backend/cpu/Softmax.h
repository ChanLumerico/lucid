// lucid/_C/backend/cpu/Softmax.h
//
// CPU softmax and log_softmax forward and backward kernels, parallelised across
// rows via cpu::parallel_for and evaluating exp in L1 tiles through Accelerate
// vForce (vvexpf / vvexp).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <Accelerate/Accelerate.h>

#include "../../core/Shape.h"
#include "Parallel.h"
#include "Vdsp.h"

namespace lucid::backend::cpu {

inline constexpr std::size_t kSoftmaxTile = 1024;
inline constexpr std::size_t kSoftmaxElemGrain = 32768;
inline constexpr std::size_t kSoftmaxStreamGrain = 65536;
inline constexpr std::size_t kSoftmaxMaxChunks = 8;

inline std::size_t
softmax_grain(std::size_t lines, std::size_t axis_dim, std::size_t elem_grain) noexcept {
    const std::size_t min_grain =
        std::max<std::size_t>(1, elem_grain / std::max<std::size_t>(axis_dim, 1));
    const std::size_t cap_grain = std::max<std::size_t>(1, lines / kSoftmaxMaxChunks);
    return std::max(min_grain, cap_grain);
}

inline void tile_exp(float* tile, std::size_t len) noexcept {
    int count = static_cast<int>(len);
    vvexpf(tile, tile, &count);
}

inline void tile_exp(double* tile, std::size_t len) noexcept {
    int count = static_cast<int>(len);
    vvexp(tile, tile, &count);
}

inline float tile_sum(const float* tile, std::size_t len) noexcept {
    return vsum_f32(tile, len);
}

inline double tile_sum(const double* tile, std::size_t len) noexcept {
    return vsum_f64(tile, len);
}

inline float tile_dot(const float* a, const float* b, std::size_t len) noexcept {
    return vdotpr_f32(a, b, len);
}

inline double tile_dot(const double* a, const double* b, std::size_t len) noexcept {
    return vdotpr_f64(a, b, len);
}

template <typename T>
inline T row_max(const T* base, std::size_t axis_dim, std::size_t stride) noexcept {
    if (axis_dim == 0)
        return -std::numeric_limits<T>::infinity();
    if constexpr (std::is_same_v<T, float>) {
        if (stride == 1)
            return vmaxval_f32(base, axis_dim);
    }
    T m = base[0];
    for (std::size_t r = 1; r < axis_dim; ++r) {
        const T v = base[r * stride];
        if (v > m || std::isnan(v))
            m = v;
    }
    return m;
}

template <typename ContigFn, typename StridedFn>
inline void dispatch_softmax_lines(std::size_t outer,
                                   std::size_t axis_dim,
                                   std::size_t inner,
                                   std::size_t elem_grain,
                                   ContigFn&& contig_fn,
                                   StridedFn&& strided_fn) {
    if (axis_dim == 0 || outer == 0 || inner == 0)
        return;
    const std::size_t lines = outer * inner;
    const std::size_t grain = softmax_grain(lines, axis_dim, elem_grain);
    if (inner == 1) {
        parallel_for(outer, grain, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = lo; o < hi; ++o)
                contig_fn(o * axis_dim);
        });
        return;
    }
    parallel_for(lines, grain, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t idx = lo; idx < hi; ++idx) {
            const std::size_t off = (idx / inner) * axis_dim * inner + (idx % inner);
            strided_fn(off);
        }
    });
}

template <typename T>
inline T
sum_shifted_exp(const T* base, T* obase, std::size_t axis_dim, std::size_t stride, T m) noexcept {
    T tile[kSoftmaxTile];
    T s = T{0};
    for (std::size_t r0 = 0; r0 < axis_dim; r0 += kSoftmaxTile) {
        const std::size_t len = std::min(kSoftmaxTile, axis_dim - r0);
        for (std::size_t k = 0; k < len; ++k)
            tile[k] = base[(r0 + k) * stride] - m;
        tile_exp(tile, len);
        if (obase) {
            for (std::size_t k = 0; k < len; ++k)
                obase[(r0 + k) * stride] = tile[k];
        }
        s += tile_sum(tile, len);
    }
    return s;
}

template <typename T>
void softmax_cpu(const T* in, T* out, std::size_t outer, std::size_t axis_dim, std::size_t inner) {
    dispatch_softmax_lines(
        outer, axis_dim, inner, kSoftmaxElemGrain,
        [&](std::size_t off) {
            const T* __restrict row = in + off;
            T* __restrict dst = out + off;
            const T m = row_max(row, axis_dim, 1);
            for (std::size_t r = 0; r < axis_dim; ++r)
                dst[r] = row[r] - m;
            tile_exp(dst, axis_dim);
            const T inv_s = T{1} / tile_sum(dst, axis_dim);
            for (std::size_t r = 0; r < axis_dim; ++r)
                dst[r] *= inv_s;
        },
        [&](std::size_t off) {
            const T* base = in + off;
            T* obase = out + off;
            const T m = row_max(base, axis_dim, inner);
            const T inv_s = T{1} / sum_shifted_exp(base, obase, axis_dim, inner, m);
            for (std::size_t r = 0; r < axis_dim; ++r)
                obase[r * inner] *= inv_s;
        });
}

template <typename T>
void log_softmax_cpu(
    const T* in, T* out, std::size_t outer, std::size_t axis_dim, std::size_t inner) {
    dispatch_softmax_lines(
        outer, axis_dim, inner, kSoftmaxElemGrain,
        [&](std::size_t off) {
            const T* __restrict row = in + off;
            T* __restrict dst = out + off;
            const T m = row_max(row, axis_dim, 1);
            const T log_sum = m + std::log(sum_shifted_exp<T>(row, nullptr, axis_dim, 1, m));
            for (std::size_t r = 0; r < axis_dim; ++r)
                dst[r] = row[r] - log_sum;
        },
        [&](std::size_t off) {
            const T* base = in + off;
            T* obase = out + off;
            const T m = row_max(base, axis_dim, inner);
            const T log_sum = m + std::log(sum_shifted_exp<T>(base, nullptr, axis_dim, inner, m));
            for (std::size_t r = 0; r < axis_dim; ++r)
                obase[r * inner] = base[r * inner] - log_sum;
        });
}

template <typename T>
void log_softmax_backward_cpu(
    const T* gy, const T* g, T* dx, std::size_t outer, std::size_t axis_dim, std::size_t inner) {
    dispatch_softmax_lines(
        outer, axis_dim, inner, kSoftmaxElemGrain,
        [&](std::size_t off) {
            const T* __restrict y_row = gy + off;
            const T* __restrict g_row = g + off;
            T* __restrict dx_row = dx + off;
            const T sum_g = tile_sum(g_row, axis_dim);
            T tile[kSoftmaxTile];
            for (std::size_t r0 = 0; r0 < axis_dim; r0 += kSoftmaxTile) {
                const std::size_t len = std::min(kSoftmaxTile, axis_dim - r0);
                for (std::size_t k = 0; k < len; ++k)
                    tile[k] = y_row[r0 + k];
                tile_exp(tile, len);
                for (std::size_t k = 0; k < len; ++k)
                    dx_row[r0 + k] = g_row[r0 + k] - tile[k] * sum_g;
            }
        },
        [&](std::size_t off) {
            const T* y_base = gy + off;
            const T* g_base = g + off;
            T* dx_base = dx + off;
            T sum_g = T{0};
            for (std::size_t r = 0; r < axis_dim; ++r)
                sum_g += g_base[r * inner];
            T tile[kSoftmaxTile];
            for (std::size_t r0 = 0; r0 < axis_dim; r0 += kSoftmaxTile) {
                const std::size_t len = std::min(kSoftmaxTile, axis_dim - r0);
                for (std::size_t k = 0; k < len; ++k)
                    tile[k] = y_base[(r0 + k) * inner];
                tile_exp(tile, len);
                for (std::size_t k = 0; k < len; ++k) {
                    const std::size_t pos = (r0 + k) * inner;
                    dx_base[pos] = g_base[pos] - tile[k] * sum_g;
                }
            }
        });
}

template <typename T>
void softmax_backward_cpu(
    const T* zp, const T* gp, T* dx, std::size_t outer, std::size_t axis_dim, std::size_t inner) {
    dispatch_softmax_lines(
        outer, axis_dim, inner, kSoftmaxStreamGrain,
        [&](std::size_t off) {
            const T* __restrict zb = zp + off;
            const T* __restrict gb = gp + off;
            T* __restrict xb = dx + off;
            const T sum = tile_dot(gb, zb, axis_dim);
            for (std::size_t r = 0; r < axis_dim; ++r)
                xb[r] = zb[r] * (gb[r] - sum);
        },
        [&](std::size_t off) {
            const T* zb = zp + off;
            const T* gb = gp + off;
            T* xb = dx + off;
            T sum = T{};
            for (std::size_t r = 0; r < axis_dim; ++r)
                sum += gb[r * inner] * zb[r * inner];
            for (std::size_t r = 0; r < axis_dim; ++r)
                xb[r * inner] = zb[r * inner] * (gb[r * inner] - sum);
        });
}

}  // namespace lucid::backend::cpu
