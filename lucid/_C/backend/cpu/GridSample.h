// lucid/_C/backend/cpu/GridSample.h
//
// ``grid_sample`` of a 4-D map on the CPU stream: bilinear or nearest, zero
// or border padding (reflection is a reflection of the grid, done before
// the kernel), with its gradients for the input and the grid.  Written once
// over the scalar type; the 16-bit formats reach it through float32.
//
// A grid coordinate is any float, NaN and infinities included, so a sample
// position is turned into an index only by ``resample::index_of``, and its
// neighbour only by ``resample::next_index``.
// The kernel used to cast the floored position to ``int`` and add one:
// past the int range the cast saturated (when it was defined at all), the
// addition overflowed, and the bounds check let a wild address through —
// x = inf or 1e10 killed the process with SIGBUS (LCD-420).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../../core/Shape.h"
#include "../SamplePositions.h"

namespace lucid::backend::cpu::grid {

enum Mode { Bilinear = 0, Nearest = 1 };
enum Padding { Zeros = 0, Border = 1 };

struct Geometry {
    std::int64_t N, C, H_in, W_in, H_out, W_out;
    int mode;
    int padding;
    bool align_corners;

    std::size_t in_plane() const noexcept { return static_cast<std::size_t>(H_in * W_in); }
    std::size_t out_plane() const noexcept { return static_cast<std::size_t>(H_out * W_out); }
};

inline Geometry geometry(
    const Shape& in_shape, const Shape& grid_shape, int mode, int padding, bool align_corners) {
    return {in_shape[0],   in_shape[1], in_shape[2], in_shape[3],  grid_shape[1],
            grid_shape[2], mode,        padding,     align_corners};
}

// A normalised grid coordinate as a position in an axis of ``size`` samples.
template <class T>
T unnormalize(T g, std::int64_t size, bool align_corners) {
    return resample::unnormalize(g, size, align_corners);
}

// d(position) / d(grid coordinate).
template <class T>
T unnormalize_slope(std::int64_t size, bool align_corners) {
    return resample::unnormalize_slope<T>(size, align_corners);
}

// One sample point: the two neighbours on each axis (the same one twice
// under nearest) and the bilinear weights of the four corners.
template <class T>
struct Taps {
    std::int64_t x0, x1, y0, y1;
    T x0f, x1f, y0f, y1f;  // the neighbours' positions, for the grid gradient
    T ix, iy;
    bool clipped_x, clipped_y;
};

template <class T>
Taps<T> taps_at(const Geometry& g, T gx, T gy) {
    Taps<T> t{};
    t.ix = unnormalize(gx, g.W_in, g.align_corners);
    t.iy = unnormalize(gy, g.H_in, g.align_corners);
    if (g.padding == Border) {
        // A position on the edge counts as clipped, as in the reference: its
        // grid gradient is zero.  NaN is neither, and passes through.
        t.clipped_x = t.ix <= T{0} || t.ix >= static_cast<T>(g.W_in - 1);
        t.clipped_y = t.iy <= T{0} || t.iy >= static_cast<T>(g.H_in - 1);
        t.ix = std::clamp(t.ix, T{0}, static_cast<T>(g.W_in - 1));
        t.iy = std::clamp(t.iy, T{0}, static_cast<T>(g.H_in - 1));
    }
    const bool nearest = g.mode == Nearest;
    t.x0f = nearest ? std::nearbyint(t.ix) : std::floor(t.ix);
    t.y0f = nearest ? std::nearbyint(t.iy) : std::floor(t.iy);
    t.x1f = t.x0f + T{1};
    t.y1f = t.y0f + T{1};
    t.x0 = resample::index_of(t.x0f);
    t.y0 = resample::index_of(t.y0f);
    t.x1 = nearest ? t.x0 : resample::next_index(t.x0);
    t.y1 = nearest ? t.y0 : resample::next_index(t.y0);
    return t;
}

// Where corner ``(y, x)`` reads in a plane, or -1 when it reads nothing (a
// zero-padded sample outside the map).  A border-padded corner past the
// edge — the far neighbour of the last sample — reads the edge.
inline std::int64_t corner(const Geometry& g, std::int64_t y, std::int64_t x) {
    const bool inside = x >= 0 && x < g.W_in && y >= 0 && y < g.H_in;
    if (!inside && g.padding == Zeros)
        return -1;
    return std::clamp<std::int64_t>(y, 0, g.H_in - 1) * g.W_in +
           std::clamp<std::int64_t>(x, 0, g.W_in - 1);
}

template <class T>
void forward(const T* x, const T* grid, T* out, const Geometry& g) {
    for (std::int64_t n = 0; n < g.N; ++n)
        for (std::int64_t p = 0; p < static_cast<std::int64_t>(g.out_plane()); ++p) {
            const T* gp = grid + (static_cast<std::size_t>(n) * g.out_plane() + p) * 2;
            const Taps<T> t = taps_at(g, gp[0], gp[1]);
            const std::int64_t at[4] = {corner(g, t.y0, t.x0), corner(g, t.y1, t.x0),
                                        corner(g, t.y0, t.x1), corner(g, t.y1, t.x1)};
            T w[4] = {T{1}, T{0}, T{0}, T{0}};
            if (g.mode == Bilinear) {
                w[0] = (t.x1f - t.ix) * (t.y1f - t.iy);
                w[1] = (t.x1f - t.ix) * (t.iy - t.y0f);
                w[2] = (t.ix - t.x0f) * (t.y1f - t.iy);
                w[3] = (t.ix - t.x0f) * (t.iy - t.y0f);
            }
            const int taps = g.mode == Bilinear ? 4 : 1;
            for (std::int64_t c = 0; c < g.C; ++c) {
                const std::size_t plane = static_cast<std::size_t>(n * g.C + c);
                const T* src = x + plane * g.in_plane();
                auto tap = [&](int k) { return (at[k] < 0 ? T{0} : src[at[k]]) * w[k]; };
                T v = tap(0);
                for (int k = 1; k < taps; ++k)
                    v += tap(k);
                out[plane * g.out_plane() + static_cast<std::size_t>(p)] = v;
            }
        }
}

// ``dx`` and ``dgrid`` are fully overwritten.
template <class T>
void backward(const T* go, const T* x, const T* grid, T* dx, T* dgrid, const Geometry& g) {
    std::memset(dx, 0, sizeof(T) * static_cast<std::size_t>(g.N * g.C) * g.in_plane());
    const T sx = unnormalize_slope<T>(g.W_in, g.align_corners);
    const T sy = unnormalize_slope<T>(g.H_in, g.align_corners);
    for (std::int64_t n = 0; n < g.N; ++n)
        for (std::int64_t p = 0; p < static_cast<std::int64_t>(g.out_plane()); ++p) {
            const std::size_t gi = (static_cast<std::size_t>(n) * g.out_plane() + p) * 2;
            const Taps<T> t = taps_at(g, grid[gi], grid[gi + 1]);
            const std::int64_t at[4] = {corner(g, t.y0, t.x0), corner(g, t.y1, t.x0),
                                        corner(g, t.y0, t.x1), corner(g, t.y1, t.x1)};
            if (g.mode == Nearest) {
                // The sample does not move with the grid: no grid gradient.
                for (std::int64_t c = 0; c < g.C && at[0] >= 0; ++c) {
                    const std::size_t plane = static_cast<std::size_t>(n * g.C + c);
                    dx[plane * g.in_plane() + at[0]] += go[plane * g.out_plane() + p];
                }
                dgrid[gi] = T{0};
                dgrid[gi + 1] = T{0};
                continue;
            }
            const T wx[2] = {t.x1f - t.ix, t.ix - t.x0f};
            const T wy[2] = {t.y1f - t.iy, t.iy - t.y0f};
            const T w[4] = {wx[0] * wy[0], wx[0] * wy[1], wx[1] * wy[0], wx[1] * wy[1]};
            T dix{0}, diy{0};
            for (std::int64_t c = 0; c < g.C; ++c) {
                const std::size_t plane = static_cast<std::size_t>(n * g.C + c);
                const T* src = x + plane * g.in_plane();
                T* dst = dx + plane * g.in_plane();
                const T gv = go[plane * g.out_plane() + p];
                T v[4];
                for (int k = 0; k < 4; ++k) {
                    v[k] = at[k] < 0 ? T{0} : src[at[k]];
                    if (at[k] >= 0)
                        dst[at[k]] += gv * w[k];
                }
                // corners: 0 = (y0, x0), 1 = (y1, x0), 2 = (y0, x1), 3 = (y1, x1)
                dix += gv * ((v[2] - v[0]) * wy[0] + (v[3] - v[1]) * wy[1]);
                diy += gv * ((v[1] - v[0]) * wx[0] + (v[3] - v[2]) * wx[1]);
            }
            // Scale by 0 when clipped so NaN from the other axis's weights
            // propagates through ``NaN * 0 = NaN`` as in the reference.
            dgrid[gi] = dix * (t.clipped_x ? T{0} : sx);
            dgrid[gi + 1] = diy * (t.clipped_y ? T{0} : sy);
        }
}

}  // namespace lucid::backend::cpu::grid
