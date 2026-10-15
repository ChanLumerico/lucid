// lucid/_C/backend/SamplePositions.h
//
// Where each output sample of ``interpolate`` reads the input, for both
// streams.
//
// The CPU kernel divided exactly (``d * in / out``) and the Metal kernel
// multiplied by ``in / out`` in float32; on a long axis the two picked
// different samples (1137 axis pairs between 1000 and 9000 long, LCD-403).
// The Metal linear kernels also built their coordinates in the input's
// dtype, so a bfloat16 resize read neighbours a whole sample away.  Now one
// rule, the reference's, decides every position, and both backends ask it.
//
// The reference computes a position in its accumulation type ``R`` —
// ``double`` for a float64 tensor, ``float`` for float32, float16 and
// bfloat16 — and floors it in ``float`` whatever ``R`` is.  So a float64
// resize lands where float32 arithmetic says, except where the product
// rounds across an integer in float but not in double.  The fraction a
// linear tap blends with stays in ``R``.
//
// Only sizes reach the resize positions: a scale factor that maps
// differently from ``in / out`` never reaches the engine kernels
// (``lucid/nn/functional/_interpolate.py``).  ``index_of`` turns any float
// position into an index, for ``grid_sample`` too.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lucid::backend::resample {

// A whole-numbered sample position ``v`` (already floored or rounded) as an
// index, whatever ``v`` is.  NaN reads as 0 and a position past the int64
// range saturates at its end — the reference reads its float positions
// this way, and so lands a NaN coordinate on sample 0.  Casting an infinite
// or out-of-range float is undefined behaviour, and the CPU grid_sample
// went on to add 1 to the saturated int: ``grid_sample`` at x = inf or
// 1e10 died with SIGBUS (LCD-420).  Callers compare the index with the
// extent before it addresses anything, and step to a neighbour with
// ``next_index``.
template <class T>
std::int64_t index_of(T v) {
    constexpr T limit = static_cast<T>(9.223372036854775808e18);  // 2^63
    if (std::isnan(v))
        return 0;
    if (v >= limit)
        return std::numeric_limits<std::int64_t>::max();
    if (v < -limit)
        return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(v);
}

// The index after ``i``, or ``i`` itself at the top of the range: a
// saturated position stays out of range instead of wrapping back into it.
inline std::int64_t next_index(std::int64_t i) {
    return i < std::numeric_limits<std::int64_t>::max() ? i + 1 : i;
}

// d(position) / d(grid coordinate): ``(size - 1) / 2`` with aligned corners,
// ``size / 2`` without.  Halving the extent before multiplying by ``g + 1``
// matches the 4-D reference kernel and avoids overflowing ``(g + 1) * size``
// when ``|g| * size`` exceeds ``float_max``.
template <class T>
T unnormalize_slope(std::int64_t size, bool align_corners) {
    return static_cast<T>(align_corners ? size - 1 : size) * static_cast<T>(0.5);
}

// A normalised grid coordinate as a position in an axis of ``size`` samples.
template <class T>
T unnormalize(T g, std::int64_t size, bool align_corners) {
    const T scaled = (g + T{1}) * unnormalize_slope<T>(size, align_corners);
    return align_corners ? scaled : scaled - static_cast<T>(0.5);
}

// The input sample output ``dst`` copies under nearest.
template <class R>
std::int64_t nearest_source(std::int64_t dst, std::int64_t in, std::int64_t out) {
    const R scale = static_cast<R>(in) / static_cast<R>(out);
    const R position = scale * static_cast<R>(dst);
    return std::min(index_of(std::floor(static_cast<float>(position))), in - 1);
}

// Output sample ``d`` blends ``lo`` and ``hi`` as ``(1 - frac) * x[lo] +
// frac * x[hi]``.  ``hi`` is ``lo`` on the last sample.
template <class R>
struct LinearTap {
    std::int64_t lo;
    std::int64_t hi;
    R frac;
};

// Distance in input samples between neighbouring output samples.
template <class R>
R linear_scale(std::int64_t in, std::int64_t out, bool align_corners) {
    if (align_corners)
        return out > 1 ? static_cast<R>(in - 1) / static_cast<R>(out - 1) : R{0};
    return static_cast<R>(in) / static_cast<R>(out);
}

// The tap of output ``dst``.  Half-pixel centres unless the corners are
// aligned, and a centre left of the first sample is clamped onto it.  An
// axis that keeps its length copies.
//
// The half-pixel centre is one fused multiply-add, rounded once, because
// the reference's is: its compiler contracts ``scale * (dst + 0.5) - 0.5``.
// Rounded twice, the centre of output 5382 of a 2155 -> 5387 resize moved
// by 1e-4 of a sample in float32, and so did its blend.
template <class R>
LinearTap<R>
linear_tap(R scale, std::int64_t dst, std::int64_t in, std::int64_t out, bool align_corners) {
    if (in == out)
        return {dst, dst, R{0}};
    R source;
    if (align_corners) {
        source = scale * static_cast<R>(dst);
    } else {
        source = std::fma(scale, static_cast<R>(dst) + R{0.5}, R{-0.5});
        if (source < R{0})
            source = R{0};
    }
    const std::int64_t lo = std::min(index_of(std::floor(static_cast<float>(source))), in - 1);
    const R frac = std::clamp(source - static_cast<R>(lo), R{0}, R{1});
    return {lo, lo < in - 1 ? lo + 1 : lo, frac};
}

// Every nearest source along an axis of ``in`` resampled to ``out``.
template <class R>
std::vector<std::int64_t> nearest_axis(std::int64_t in, std::int64_t out) {
    std::vector<std::int64_t> sources(static_cast<std::size_t>(out));
    for (std::int64_t d = 0; d < out; ++d)
        sources[static_cast<std::size_t>(d)] = nearest_source<R>(d, in, out);
    return sources;
}

// Every linear tap along an axis of ``in`` resampled to ``out``.
template <class R>
std::vector<LinearTap<R>> linear_axis(std::int64_t in, std::int64_t out, bool align_corners) {
    const R scale = linear_scale<R>(in, out, align_corners);
    std::vector<LinearTap<R>> taps(static_cast<std::size_t>(out));
    for (std::int64_t d = 0; d < out; ++d)
        taps[static_cast<std::size_t>(d)] = linear_tap<R>(scale, d, in, out, align_corners);
    return taps;
}

}  // namespace lucid::backend::resample
