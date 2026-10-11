// lucid/_C/backend/gpu/SortOrder.h
//
// The order ``sort``, ``argsort`` and ``topk`` read a tensor in on the MLX
// (Metal) stream.  MLX's ``argsort`` is stable and ascending, NaN last.
//
// ``topk`` wants the largest first, and among equal values the lowest
// index first, as the CPU and the reference give it.  Reading the stable
// ascending order backwards gave equal values highest index first
// (``topk([0, 5, 4, 4], 2)`` named 1 and 3 on Metal, 1 and 2 on the CPU;
// LCD-407).  So the axis is reversed first: the stable ascending order of
// the reversed axis lists equal values highest original index first, and
// read backwards it lists them lowest first.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

namespace lucid {
namespace backend {

// ``argsort`` that also takes bool.  MLX ships no bool sort kernel —
// sorting a bool tensor on Metal died with "Unable to load kernel
// carg_block_sort_bool_…" — so the order is taken on uint8, where false <
// true exactly as for bool, and the caller gathers the original values
// with it.
inline ::mlx::core::array gpu_sort_order(const ::mlx::core::array& a, int axis) {
    if (a.dtype() == ::mlx::core::bool_)
        return ::mlx::core::argsort(::mlx::core::astype(a, ::mlx::core::uint8), axis);
    return ::mlx::core::argsort(a, axis);
}

// Positions ``first, first - 1, ...``, ``count`` of them, as int32.
inline ::mlx::core::array countdown(int first, int count) {
    std::vector<std::int32_t> positions(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
        positions[static_cast<std::size_t>(i)] = first - i;
    return ::mlx::core::array(positions.data(), ::mlx::core::Shape{count}, ::mlx::core::int32);
}

// The positions of the ``k`` largest values along ``axis``, largest first,
// equal values lowest index first and NaN, which sorts as the largest,
// before every number.
inline ::mlx::core::array gpu_descending_order(const ::mlx::core::array& a, int axis, int k) {
    namespace mx = ::mlx::core;
    const int last = a.shape(axis) - 1;
    const auto nd = static_cast<std::size_t>(a.ndim());
    mx::Shape start(nd, 0), stop = a.shape(), step(nd, 1);
    start[static_cast<std::size_t>(axis)] = last;
    stop[static_cast<std::size_t>(axis)] = -last - 2;  // one before index 0
    step[static_cast<std::size_t>(axis)] = -1;
    const auto reversed = mx::slice(a, start, stop, step);
    const auto order = mx::take(gpu_sort_order(reversed, axis), countdown(last, k), axis);
    return mx::subtract(mx::array(last, order.dtype()), order);
}

}  // namespace backend
}  // namespace lucid
