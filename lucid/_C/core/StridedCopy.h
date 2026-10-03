// lucid/_C/core/StridedCopy.h
//
// Moving the elements of a strided CPU view to and from a dense, row-major
// buffer: the copy behind ``contiguous()``, behind every op that reads a
// view (TensorImpl::storage), and behind a write through a view.
//
// Both directions used to walk element by element — a coordinate step and
// a memcpy per element — which made a 1024 x 1024 float transpose take
// 4.6 ms to copy where the same bytes copy in 0.08.  Here the axes are
// merged first, and the innermost run moves as one memcpy when it is
// contiguous, as a fill when it is broadcast, and as a typed strided loop
// otherwise.  A view that merges to a plain 2-D transpose is copied in
// cache-sized tiles.  Strides are in bytes and non-negative.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "Shape.h"

namespace lucid::strided {

// A strided view laid out as (outer..., inner) blocks after merging axes:
// size-1 axes dropped, and an axis folded into the one before it whenever
// the outer one steps exactly over the inner one's extent.  Strides are in
// bytes.  ``coalesce`` is what lets a slice of whole rows, or an expanded
// axis next to another, become one long run.
struct Coalesced {
    std::vector<std::int64_t> extent;
    std::vector<std::int64_t> stride;
};

inline Coalesced coalesce(const Shape& shape, const Stride& stride) {
    Coalesced c;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] == 1)
            continue;
        if (!c.extent.empty() && c.stride.back() == stride[d] * shape[d]) {
            c.extent.back() *= shape[d];
            c.stride.back() = stride[d];
        } else {
            c.extent.push_back(shape[d]);
            c.stride.push_back(stride[d]);
        }
    }
    return c;
}

// One element of ``elem`` bytes as an unsigned integer of that width, so a
// run moves by typed loads and stores rather than a memcpy per element.
template <class F>
inline bool with_element_type(std::size_t elem, F&& f) {
    switch (elem) {
    case 1:
        f(std::uint8_t{});
        return true;
    case 2:
        f(std::uint16_t{});
        return true;
    case 4:
        f(std::uint32_t{});
        return true;
    case 8:
        f(std::uint64_t{});
        return true;
    default:
        return false;  // 16-byte complex: the caller copies element by element
    }
}

// Packs a strided view into ``dst``, dense and row-major.  ``src`` points at
// the view's first element.
//
// This walked every element recursively with a memcpy each — 4.6 ms to
// make a 1024 x 1024 float transpose contiguous, where copying the same
// bytes takes 0.08 — and every CPU op that reads a view comes through here.
// Now the axes are merged first, and the innermost run moves as a memcpy
// when it is contiguous, a fill when it is broadcast, and a typed strided
// loop otherwise; a view that merges to a 2-D transpose is copied in tiles,
// so both sides stay in cache.
inline void pack(const std::byte* src,
                 std::byte* dst,
                 const Shape& shape,
                 const Stride& stride,
                 std::size_t elem) {
    const Coalesced c = coalesce(shape, stride);
    if (c.extent.empty()) {
        std::memcpy(dst, src, elem);
        return;
    }
    const std::size_t ndim = c.extent.size();
    const std::int64_t inner = c.extent.back();
    const std::int64_t step = c.stride.back();

    // A transpose: the inner axis jumps, the outer one is contiguous.
    if (ndim == 2 && c.stride[0] == static_cast<std::int64_t>(elem) &&
        step != static_cast<std::int64_t>(elem) && step != 0) {
        const std::int64_t rows = c.extent[0];
        constexpr std::int64_t kTile = 32;
        const bool typed = with_element_type(elem, [&](auto zero) {
            using T = decltype(zero);
            T* out = reinterpret_cast<T*>(dst);
            for (std::int64_t r0 = 0; r0 < rows; r0 += kTile)
                for (std::int64_t c0 = 0; c0 < inner; c0 += kTile) {
                    const std::int64_t r1 = std::min(rows, r0 + kTile);
                    const std::int64_t c1 = std::min(inner, c0 + kTile);
                    for (std::int64_t r = r0; r < r1; ++r)
                        for (std::int64_t col = c0; col < c1; ++col) {
                            T v;
                            std::memcpy(&v, src + r * elem + col * step, sizeof(T));
                            out[r * inner + col] = v;
                        }
                }
        });
        if (typed)
            return;
    }

    auto run = [&](const std::byte* from, std::byte* to) {
        if (step == static_cast<std::int64_t>(elem)) {
            std::memcpy(to, from, static_cast<std::size_t>(inner) * elem);
            return;
        }
        const bool typed = with_element_type(elem, [&](auto zero) {
            using T = decltype(zero);
            T* out = reinterpret_cast<T*>(to);
            if (step == 0) {
                T v;
                std::memcpy(&v, from, sizeof(T));
                std::fill(out, out + inner, v);
                return;
            }
            for (std::int64_t i = 0; i < inner; ++i) {
                T v;
                std::memcpy(&v, from + i * step, sizeof(T));
                out[i] = v;
            }
        });
        if (!typed)
            for (std::int64_t i = 0; i < inner; ++i)
                std::memcpy(to + static_cast<std::size_t>(i) * elem, from + i * step, elem);
    };

    // The outer axes as an odometer, the source offset kept incrementally.
    std::vector<std::int64_t> idx(ndim - 1, 0);
    std::int64_t from = 0;
    const std::size_t run_bytes = static_cast<std::size_t>(inner) * elem;
    std::size_t outer = 1;
    for (std::size_t d = 0; d + 1 < ndim; ++d)
        outer *= static_cast<std::size_t>(c.extent[d]);
    for (std::size_t r = 0; r < outer; ++r) {
        run(src + from, dst + r * run_bytes);
        for (std::size_t d = ndim - 1; d-- > 0;) {
            from += c.stride[d];
            if (++idx[d] < c.extent[d])
                break;
            from -= c.stride[d] * c.extent[d];
            idx[d] = 0;
        }
    }
}

// Lays ``packed`` — ``shape``'s elements, dense and row-major — into the
// view at ``first`` through ``stride``: the inverse of :func:`pack`.  Runs
// move as a memcpy when the inner axis is contiguous and as typed stores
// otherwise.  The view must not repeat an element (``overlaps_itself``).
inline void scatter(const std::byte* packed,
                    std::byte* first,
                    const Shape& shape,
                    const Stride& stride,
                    std::size_t elem) {
    const Coalesced c = coalesce(shape, stride);
    if (c.extent.empty()) {
        std::memcpy(first, packed, elem);
        return;
    }
    const std::size_t ndim = c.extent.size();
    const std::int64_t inner = c.extent.back();
    const std::int64_t step = c.stride.back();
    auto run = [&](const std::byte* from, std::byte* to) {
        if (step == static_cast<std::int64_t>(elem)) {
            std::memcpy(to, from, static_cast<std::size_t>(inner) * elem);
            return;
        }
        const bool typed = with_element_type(elem, [&](auto zero) {
            using T = decltype(zero);
            const T* in = reinterpret_cast<const T*>(from);
            for (std::int64_t i = 0; i < inner; ++i)
                std::memcpy(to + i * step, &in[i], sizeof(T));
        });
        if (!typed)
            for (std::int64_t i = 0; i < inner; ++i)
                std::memcpy(to + i * step, from + static_cast<std::size_t>(i) * elem, elem);
    };
    std::vector<std::int64_t> idx(ndim - 1, 0);
    std::int64_t at = 0;
    const std::size_t run_bytes = static_cast<std::size_t>(inner) * elem;
    std::size_t outer = 1;
    for (std::size_t d = 0; d + 1 < ndim; ++d)
        outer *= static_cast<std::size_t>(c.extent[d]);
    for (std::size_t r = 0; r < outer; ++r) {
        run(packed + r * run_bytes, first + at);
        for (std::size_t d = ndim - 1; d-- > 0;) {
            at += c.stride[d];
            if (++idx[d] < c.extent[d])
                break;
            at -= c.stride[d] * c.extent[d];
            idx[d] = 0;
        }
    }
}

}  // namespace lucid::strided
