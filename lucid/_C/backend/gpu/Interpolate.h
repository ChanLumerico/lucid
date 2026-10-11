// lucid/_C/backend/gpu/Interpolate.h
//
// ``interpolate`` on the MLX (Metal) stream: nearest and (bi/tri)linear
// resizes of 2-D and 3-D maps and their adjoints.
//
// Where every output sample reads is decided on the host by
// ``SamplePositions.h``, the rule the CPU kernels follow too, and reaches
// the graph as an index and a weight per output position of each axis.
// The kernel used to build those positions in the graph — in float32 for
// nearest, which picked different samples from the CPU's exact division on
// a long axis, and in the input's own dtype for the linear modes, which
// moved a bfloat16 sample by a whole neighbour (LCD-403).
//
// Each resampled axis is applied in turn, the last first, as the reference
// nests its sum.  A linear axis reads both taps of every position, a zero
// weight included, so an infinite neighbour gives NaN as it does there.
// Float16 and bfloat16 blend and accumulate in float32 and round once, as
// the CPU stream does.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../SamplePositions.h"
#include "HalfAccumulation.h"
#include "MlxBridge.h"

namespace lucid::gpu::interp {

namespace mx = ::mlx::core;

enum class Filter { Nearest, Linear };

// ``v`` laid along ``axis`` of an ``ndim``-axis array, length one elsewhere.
inline mx::array along(const mx::array& v, int axis, int ndim) {
    mx::Shape shape(static_cast<std::size_t>(ndim), 1);
    shape[static_cast<std::size_t>(axis)] = v.shape(0);
    return mx::reshape(v, shape);
}

inline mx::array int32_array(const std::vector<std::int32_t>& v) {
    return mx::array(v.data(), mx::Shape{static_cast<int>(v.size())}, mx::int32);
}

inline mx::array float_array(const std::vector<float>& v) {
    return mx::array(v.data(), mx::Shape{static_cast<int>(v.size())}, mx::float32);
}

inline mx::array nearest_index(std::int64_t in, std::int64_t out) {
    const auto sources = backend::resample::nearest_axis<float>(in, out);
    return int32_array(std::vector<std::int32_t>(sources.begin(), sources.end()));
}

// The two taps of every output position along one axis, and their weights.
struct LinearAxis {
    mx::array lo, hi, keep, frac;
};

inline LinearAxis linear_axis(std::int64_t in, std::int64_t out, bool align_corners) {
    const auto taps = backend::resample::linear_axis<float>(in, out, align_corners);
    std::vector<std::int32_t> lo, hi;
    std::vector<float> keep, frac;
    for (const auto& t : taps) {
        lo.push_back(static_cast<std::int32_t>(t.lo));
        hi.push_back(static_cast<std::int32_t>(t.hi));
        keep.push_back(1.0f - t.frac);
        frac.push_back(t.frac);
    }
    return {int32_array(lo), int32_array(hi), float_array(keep), float_array(frac)};
}

// ``g`` summed onto ``in`` samples along ``axis``: every element goes to
// ``idx`` of its position.
inline mx::array scatter_back(const mx::array& g, const mx::array& idx, int axis, int in) {
    auto shape = g.shape();
    shape[static_cast<std::size_t>(axis)] = in;
    auto base = mx::zeros(shape, g.dtype());
    if (g.size() == 0 || base.size() == 0)
        return base;
    const int ndim = static_cast<int>(g.ndim());
    return mx::scatter_add_axis(base, mx::broadcast_to(along(idx, axis, ndim), g.shape()), g, axis);
}

// One axis of ``x`` resampled to ``out`` (or, ``backward``, a gradient
// ``x`` of that taken back to ``in`` samples).
inline mx::array resample_axis(const mx::array& x,
                               Filter filter,
                               int axis,
                               std::int64_t in,
                               std::int64_t out,
                               bool align_corners,
                               bool backward) {
    const int ndim = static_cast<int>(x.ndim());
    if (filter == Filter::Nearest) {
        const auto idx = nearest_index(in, out);
        return backward ? scatter_back(x, idx, axis, static_cast<int>(in)) : mx::take(x, idx, axis);
    }
    const auto t = linear_axis(in, out, align_corners);
    const auto keep = along(t.keep, axis, ndim);
    const auto frac = along(t.frac, axis, ndim);
    if (backward)
        return mx::add(scatter_back(mx::multiply(x, keep), t.lo, axis, static_cast<int>(in)),
                       scatter_back(mx::multiply(x, frac), t.hi, axis, static_cast<int>(in)));
    return mx::add(mx::multiply(mx::take(x, t.lo, axis), keep),
                   mx::multiply(mx::take(x, t.hi, axis), frac));
}

// ``src`` of ``(N, C, *in)`` resized to ``(N, C, *out)``, or, ``backward``,
// a gradient of ``(N, C, *out)`` taken back to ``(N, C, *in)``.  ``in_shape``
// is the input's shape either way.
inline Storage resample(const Storage& src,
                        const Shape& in_shape,
                        const std::vector<std::int64_t>& out,
                        Filter filter,
                        bool align_corners,
                        Dtype dt,
                        bool backward) {
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    // A copy needs no widening; a sum does.
    const bool widen = filter == Filter::Linear || backward;
    auto x = *std::get<GpuStorage>(src).arr;
    if (widen)
        x = backend::widen_half(x);
    const int first = static_cast<int>(in_shape.size() - out.size());
    const int rank = static_cast<int>(out.size());
    for (int k = rank - 1; k >= 0; --k) {
        const int axis = backward ? first + (rank - 1 - k) : first + k;
        const auto j = static_cast<std::size_t>(axis - first);
        x = resample_axis(x, filter, axis, in_shape[static_cast<std::size_t>(axis)], out[j],
                          align_corners, backward);
    }
    return Storage{gpu::wrap_mlx_array(backend::narrow_to(x, mlx_dt), dt)};
}

}  // namespace lucid::gpu::interp
