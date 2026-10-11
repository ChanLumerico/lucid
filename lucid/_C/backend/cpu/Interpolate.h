// lucid/_C/backend/cpu/Interpolate.h
//
// ``interpolate`` on the CPU stream: nearest and (bi/tri)linear resizes of
// 2-D and 3-D maps and their adjoints, written once over the scalar type.
// The two 16-bit formats reach these through float32 (``resample`` below).
//
// A 2-D map is a 3-D one a single sample deep, but its depth axis is never
// blended: the reference blends only the axes the mode names, and blending
// ``x * 1 + x * 0`` would turn an infinite sample into NaN.
//
// Where every output sample reads is decided by ``SamplePositions.h``, the
// rule the Metal kernels follow too.  Each linear axis is blended in turn,
// the last first, as the reference nests its sum — every tap is read, a
// zero weight included.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../IBackend.h"
#include "../SamplePositions.h"
#include "Parallel.h"

namespace lucid::backend::cpu::interp {

// Extents of the resampled axes.  A 2-D map has ``d == 1``.
struct Volume {
    std::int64_t d;
    std::int64_t h;
    std::int64_t w;

    std::size_t size() const noexcept { return static_cast<std::size_t>(d * h * w); }
};

enum class Filter { Nearest, Linear };

// One resize: ``planes`` maps (batch times channels) of ``in`` to ``out``.
struct Request {
    Filter filter;
    int rank;  // 2 or 3 resampled axes
    bool align_corners;
    std::size_t planes;
    Volume in;
    Volume out;
};

// ``in_shape`` is ``(N, C, H, W)`` or ``(N, C, D, H, W)``.
inline Request make_request(Filter filter,
                            const Shape& in_shape,
                            std::int64_t d_out,
                            std::int64_t h_out,
                            std::int64_t w_out,
                            bool align_corners) {
    const bool deep = in_shape.size() == 5;
    const std::int64_t d_in = deep ? in_shape[2] : 1;
    return {filter,
            deep ? 3 : 2,
            align_corners,
            static_cast<std::size_t>(in_shape[0] * in_shape[1]),
            {d_in, in_shape[in_shape.size() - 2], in_shape[in_shape.size() - 1]},
            {d_out, h_out, w_out}};
}

// The taps of every output position along each axis.
template <class R>
struct LinearPlan {
    std::vector<resample::LinearTap<R>> z, y, x;

    explicit LinearPlan(const Request& r)
        : z(resample::linear_axis<R>(r.in.d, r.out.d, r.align_corners)),
          y(resample::linear_axis<R>(r.in.h, r.out.h, r.align_corners)),
          x(resample::linear_axis<R>(r.in.w, r.out.w, r.align_corners)) {}
};

// Planes per parallel chunk: enough output samples to be worth a core.
inline std::size_t plane_grain(const Request& r) {
    const std::size_t per_plane = std::max(r.out.size(), r.in.size());
    return std::max<std::size_t>(1, 32768 / std::max<std::size_t>(per_plane, 1));
}

template <class T>
void nearest_forward(const T* x, T* y, const Request& r) {
    const auto zs = resample::nearest_axis<T>(r.in.d, r.out.d);
    const auto ys = resample::nearest_axis<T>(r.in.h, r.out.h);
    const auto xs = resample::nearest_axis<T>(r.in.w, r.out.w);
    parallel_for(r.planes, plane_grain(r), [&](std::size_t begin, std::size_t end) {
        for (std::size_t p = begin; p < end; ++p) {
            const T* src = x + p * r.in.size();
            T* dst = y + p * r.out.size();
            for (std::int64_t d = 0; d < r.out.d; ++d)
                for (std::int64_t h = 0; h < r.out.h; ++h) {
                    const T* row = src + (zs[d] * r.in.h + ys[h]) * r.in.w;
                    T* out_row = dst + (d * r.out.h + h) * r.out.w;
                    for (std::int64_t w = 0; w < r.out.w; ++w)
                        out_row[w] = row[xs[w]];
                }
        }
    });
}

// ``dx`` is fully overwritten.
template <class T>
void nearest_backward(const T* g, T* dx, const Request& r) {
    const auto zs = resample::nearest_axis<T>(r.in.d, r.out.d);
    const auto ys = resample::nearest_axis<T>(r.in.h, r.out.h);
    const auto xs = resample::nearest_axis<T>(r.in.w, r.out.w);
    parallel_for(r.planes, plane_grain(r), [&](std::size_t begin, std::size_t end) {
        for (std::size_t p = begin; p < end; ++p) {
            const T* src = g + p * r.out.size();
            T* dst = dx + p * r.in.size();
            std::memset(dst, 0, sizeof(T) * r.in.size());
            for (std::int64_t d = 0; d < r.out.d; ++d)
                for (std::int64_t h = 0; h < r.out.h; ++h) {
                    T* row = dst + (zs[d] * r.in.h + ys[h]) * r.in.w;
                    const T* g_row = src + (d * r.out.h + h) * r.out.w;
                    for (std::int64_t w = 0; w < r.out.w; ++w)
                        row[xs[w]] += g_row[w];
                }
        }
    });
}

// One output row blended along the last axis from input row ``row``.
template <class T>
inline void blend_row(const T* row, T* out, const std::vector<resample::LinearTap<T>>& xs) {
    for (std::size_t w = 0; w < xs.size(); ++w) {
        const auto& t = xs[w];
        out[w] = row[t.lo] * (T{1} - t.frac) + row[t.hi] * t.frac;
    }
}

// ``out = a * (1 - frac) + b * frac``, element by element.
template <class T>
inline void blend_rows(const T* a, const T* b, T frac, T* out, std::int64_t n) {
    const T keep = T{1} - frac;
    for (std::int64_t i = 0; i < n; ++i)
        out[i] = a[i] * keep + b[i] * frac;
}

template <class T>
void linear_forward(const T* x, T* y, const Request& r) {
    const LinearPlan<T> plan(r);
    const std::int64_t ow = r.out.w;
    parallel_for(r.planes, plane_grain(r), [&](std::size_t begin, std::size_t end) {
        // Rows blended along the last axis, then along the middle one; the
        // depth axis (3-D only) blends two of those.
        std::vector<T> lo_row(static_cast<std::size_t>(ow)), hi_row(lo_row.size());
        std::vector<T> plane_lo(static_cast<std::size_t>(r.out.h * ow));
        std::vector<T> plane_hi(plane_lo.size());
        auto blend_plane = [&](const T* src, std::int64_t z, T* out) {
            for (std::int64_t h = 0; h < r.out.h; ++h) {
                const auto& t = plan.y[static_cast<std::size_t>(h)];
                blend_row(src + (z * r.in.h + t.lo) * r.in.w, lo_row.data(), plan.x);
                blend_row(src + (z * r.in.h + t.hi) * r.in.w, hi_row.data(), plan.x);
                blend_rows(lo_row.data(), hi_row.data(), t.frac, out + h * ow, ow);
            }
        };
        for (std::size_t p = begin; p < end; ++p) {
            const T* src = x + p * r.in.size();
            T* dst = y + p * r.out.size();
            if (r.rank == 2) {
                blend_plane(src, 0, dst);
                continue;
            }
            for (std::int64_t d = 0; d < r.out.d; ++d) {
                const auto& t = plan.z[static_cast<std::size_t>(d)];
                blend_plane(src, t.lo, plane_lo.data());
                blend_plane(src, t.hi, plane_hi.data());
                blend_rows(plane_lo.data(), plane_hi.data(), t.frac, dst + d * r.out.h * ow,
                           r.out.h * ow);
            }
        }
    });
}

// ``dx`` is fully overwritten.  Each gradient sample goes to every tap it
// was read from, weighted as it was read.
template <class T>
void linear_backward(const T* g, T* dx, const Request& r) {
    const LinearPlan<T> plan(r);
    const bool deep = r.rank == 3;
    parallel_for(r.planes, plane_grain(r), [&](std::size_t begin, std::size_t end) {
        for (std::size_t p = begin; p < end; ++p) {
            const T* src = g + p * r.out.size();
            T* dst = dx + p * r.in.size();
            std::memset(dst, 0, sizeof(T) * r.in.size());
            for (std::int64_t d = 0; d < r.out.d; ++d) {
                const auto& tz = plan.z[static_cast<std::size_t>(d)];
                const std::int64_t zs[2] = {deep ? tz.lo : 0, deep ? tz.hi : 0};
                const T wz[2] = {deep ? T{1} - tz.frac : T{1}, deep ? tz.frac : T{0}};
                const int z_taps = deep ? 2 : 1;
                for (std::int64_t h = 0; h < r.out.h; ++h) {
                    const auto& ty = plan.y[static_cast<std::size_t>(h)];
                    const std::int64_t ys[2] = {ty.lo, ty.hi};
                    const T wy[2] = {T{1} - ty.frac, ty.frac};
                    const T* g_row = src + (d * r.out.h + h) * r.out.w;
                    for (int a = 0; a < z_taps; ++a)
                        for (int b = 0; b < 2; ++b) {
                            T* row = dst + (zs[a] * r.in.h + ys[b]) * r.in.w;
                            const T wzy = wz[a] * wy[b];
                            for (std::int64_t w = 0; w < r.out.w; ++w) {
                                const auto& tx = plan.x[static_cast<std::size_t>(w)];
                                const T gw = g_row[w] * wzy;
                                row[tx.lo] += gw * (T{1} - tx.frac);
                                row[tx.hi] += gw * tx.frac;
                            }
                        }
                }
            }
        }
    });
}

// ``src`` resized as ``r`` asks (or, ``backward``, a gradient of the resized
// map taken back to the input), in ``dt``, float32 or float64.
template <class T>
CpuStorage run(const CpuStorage& src, const Request& r, Dtype dt, bool backward) {
    const std::size_t n = r.planes * (backward ? r.in.size() : r.out.size());
    CpuStorage out{allocate_aligned_bytes(n * sizeof(T), Device::CPU), n * sizeof(T), dt};
    const T* s = reinterpret_cast<const T*>(src.ptr.get());
    T* o = reinterpret_cast<T*>(out.ptr.get());
    const bool nearest = r.filter == Filter::Nearest;
    if (backward && nearest)
        nearest_backward(s, o, r);
    else if (backward)
        linear_backward(s, o, r);
    else if (nearest)
        nearest_forward(s, o, r);
    else
        linear_forward(s, o, r);
    return out;
}

// ``src`` resized as ``r`` asks (or, ``backward``, a gradient taken back),
// in ``dt``: float32 and float64 directly, float16 and bfloat16 through
// float32 by ``be``'s ``astype``, as the rest of the CPU backend widens them.
inline Storage
resample(IBackend& be, const Storage& src, const Request& r, Dtype dt, bool backward) {
    if (is_half_float(dt)) {
        const auto n_in = static_cast<std::int64_t>(r.planes * (backward ? r.out : r.in).size());
        const auto n_out = static_cast<std::int64_t>(r.planes * (backward ? r.in : r.out).size());
        Storage wide = be.astype(src, Shape{n_in}, dt, Dtype::F32);
        return be.astype(resample(be, wide, r, Dtype::F32, backward), Shape{n_out}, Dtype::F32, dt);
    }
    const auto& cs = std::get<CpuStorage>(src);
    if (dt == Dtype::F32)
        return Storage{run<float>(cs, r, dt, backward)};
    if (dt == Dtype::F64)
        return Storage{run<double>(cs, r, dt, backward)};
    ErrorBuilder("cpu::interpolate").not_implemented("dtype must be a floating dtype");
}

}  // namespace lucid::backend::cpu::interp
