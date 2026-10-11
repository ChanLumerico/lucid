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
// Only sizes reach here: a scale factor that maps differently from
// ``in / out`` never reaches the engine kernels
// (``lucid/nn/functional/_interpolate.py``).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lucid::backend::resample {

// The input sample output ``dst`` copies under nearest.
template <class R>
std::int64_t nearest_source(std::int64_t dst, std::int64_t in, std::int64_t out) {
    const R scale = static_cast<R>(in) / static_cast<R>(out);
    const R position = scale * static_cast<R>(dst);
    const auto index = static_cast<std::int64_t>(std::floor(static_cast<float>(position)));
    return std::min(index, in - 1);
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
    const auto floored = static_cast<std::int64_t>(std::floor(static_cast<float>(source)));
    const std::int64_t lo = std::min(floored, in - 1);
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
