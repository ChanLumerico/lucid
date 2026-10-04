// lucid/_C/backend/gpu/AxisIndex.h
//
// Where a caller's index values meet an MLX gather or scatter on the GPU
// stream, and the one place that keeps them inside the buffer.
//
// MLX's Metal gather and scatter kernels (``gather.h``, ``gather_axis.h``,
// ``scatter.h`` and ``scatter_axis.h`` in MLX 0.32) add the axis length to a
// negative index once and then use the value as an offset.  Nothing compares
// the index with the axis.  So an out-of-range index read GPU memory outside
// the source (``gather(arange(4), 0, [64])`` answered 7.3e28), and an
// out-of-range scatter wrote outside its buffer.
//
// The CPU kernels refuse such an index with IndexError.  For Metal to refuse
// it, the indices would have to come back to the host: a pipeline stall on
// every gather, scatter and embedding.  So Metal isolates the bad index
// inside the graph instead.  This is policy B, decided by the user on
// 2026-10-05 (LCD-228):
//
//   * a gather answers NaN at an out-of-range position, or 0 when the result
//     is an integer or bool (those have no NaN);
//   * a scatter drops an out-of-range update, and the base keeps its value
//     there;
//   * no index outside the axis ever reaches an MLX kernel.
//
// Every ``GpuBackend`` entry point that hands a caller's index values to an
// MLX gather or scatter goes through ``gpu_axis_index``, usually by way of
// ``gpu_take_along_axis``, ``gpu_take``, ``gpu_scatter_reduce_axis`` or
// ``gpu_sink_index``.  The device difference is documented in
// lucid/test/audit/README.md and pinned by
// lucid/test/unit/ops/test_index_bounds_contract.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/ops.h>
#include <mlx/utils.h>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"

namespace lucid {
namespace backend {

// An axis scatter over the corner of ``base`` that ``idx`` covers.
//
// The index may be shorter than ``base`` on every axis but ``dim`` and
// shorter than ``src`` on every axis — the reference's rule, and the CPU's.
// MLX's axis scatters want all three to agree off ``dim`` and raised
// "[broadcast_shapes] Shapes (4) and (2) cannot be broadcast" for a valid
// call.  So ``src`` is cut to the index's extent, the scatter runs on the
// matching corner of ``base``, and the corner is written back.
template <class Scatter>
::mlx::core::array scatter_on_index_corner(const ::mlx::core::array& base,
                                           const ::mlx::core::array& idx,
                                           const ::mlx::core::array& src,
                                           int dim,
                                           Scatter&& scatter) {
    const int ndim = static_cast<int>(base.ndim());
    ::mlx::core::Shape start(static_cast<std::size_t>(ndim), 0);
    ::mlx::core::Shape src_stop = idx.shape();
    ::mlx::core::Shape corner = idx.shape();
    corner[static_cast<std::size_t>(dim)] = base.shape(dim);
    const bool cut_src = src.shape() != src_stop;
    const bool cut_base = base.shape() != corner;
    auto updates = cut_src ? ::mlx::core::slice(src, start, src_stop) : src;
    if (!cut_base)
        return scatter(base, idx, updates);
    auto region = ::mlx::core::slice(base, start, corner);
    return ::mlx::core::slice_update(base, scatter(region, idx, updates), start, corner);
}

// What an index below zero means.
enum class NegativeIndex {
    Wrap,        // ``-k`` names ``extent - k``, once: gather, scatter, take
    OutOfRange,  // nothing lies below zero: a table row, a class
};

// An index that is safe to hand to an MLX gather or scatter.
struct AxisIndex {
    // In [0, extent) everywhere, and the index itself where it is in range.
    ::mlx::core::array safe;
    // Bool, the index's shape: whether the index named a real position.
    ::mlx::core::array in_range;
};

namespace axis_index_detail {

using Fused =
    std::function<std::vector<::mlx::core::array>(const std::vector<::mlx::core::array>&)>;

// The integer width an index is checked at.  It must hold every value of
// the index and ``extent`` as well.  A narrower index is widened first,
// because ``idx + extent`` wraps in int8.  An int64 index is never narrowed,
// because narrowing first is how an index of 2^32 + 1 read row 1.
inline ::mlx::core::Dtype work_dtype(::mlx::core::Dtype dt, std::int64_t extent) {
    namespace mx = ::mlx::core;
    if (dt == mx::int64 || dt == mx::uint64)
        return dt;
    if (dt == mx::uint32 || extent > std::numeric_limits<std::int32_t>::max())
        return mx::int64;
    return mx::int32;
}

// ``in[0]`` is the index at its work width and ``in[1]`` is ``extent`` at
// the same width.  ``remainder`` is floored, so it falls in [0, extent) for
// every index.  Where the index is in range, that is the index itself, or
// the index plus extent when the index is negative.  ``Narrow`` hands the
// safe index back as int32, which holds it whenever ``extent`` fits: an
// int64 index is checked at its own width, but the gather that follows then
// reads half the bytes.
template <bool WrapNegative, bool Narrow>
std::vector<::mlx::core::array> check(const std::vector<::mlx::core::array>& in) {
    namespace mx = ::mlx::core;
    const auto& i = in[0];
    const auto& n = in[1];
    auto safe = mx::remainder(i, n);
    if (Narrow && safe.dtype() != mx::int32)
        safe = mx::astype(safe, mx::int32);
    if (mx::issubdtype(i.dtype(), mx::unsignedinteger))
        return {safe, mx::less(i, n)};
    auto lowest = WrapNegative ? mx::negative(n) : mx::zeros_like(n);
    return {safe, mx::logical_and(mx::greater_equal(i, lowest), mx::less(i, n))};
}

// One Metal kernel per call instead of one per elementwise step: an index
// is often as large as the gather it feeds, and each separate step would
// be another pass over it.
inline const Fused& fused(NegativeIndex negative, bool narrow) {
    static const Fused wrap = ::mlx::core::compile(&check<true, false>, /*shapeless=*/true);
    static const Fused wrap32 = ::mlx::core::compile(&check<true, true>, /*shapeless=*/true);
    static const Fused strict = ::mlx::core::compile(&check<false, false>, /*shapeless=*/true);
    static const Fused strict32 = ::mlx::core::compile(&check<false, true>, /*shapeless=*/true);
    if (negative == NegativeIndex::Wrap)
        return narrow ? wrap32 : wrap;
    return narrow ? strict32 : strict;
}

}  // namespace axis_index_detail

// ``idx`` checked against an axis of ``extent`` positions.  ``extent`` must
// be positive.  An empty axis has no position to point at, so every caller
// answers for it without indexing (an all-fill gather, a scatter that
// leaves the base alone).
inline AxisIndex gpu_axis_index(const ::mlx::core::array& idx,
                                std::int64_t extent,
                                NegativeIndex negative = NegativeIndex::Wrap) {
    namespace mx = ::mlx::core;
    if (!mx::issubdtype(idx.dtype(), mx::integer) && idx.dtype() != mx::bool_) {
        std::ostringstream got;
        got << idx.dtype();
        throw DtypeMismatch("an integer dtype", got.str(),
                            "gpu_axis_index: indices must be an integer tensor");
    }
    if (extent <= 0)
        ErrorBuilder("gpu_axis_index").fail("an empty axis has no position to index");
    const auto work = axis_index_detail::work_dtype(idx.dtype(), extent);
    auto i = idx.dtype() == work ? idx : mx::astype(idx, work);
    const bool narrow = extent <= std::numeric_limits<std::int32_t>::max();
    auto out = axis_index_detail::fused(negative, narrow)({std::move(i), mx::array(extent, work)});
    return {std::move(out[0]), std::move(out[1])};
}

// What a gather answers at an out-of-range position.  A floating or complex
// result gets NaN, which no real lookup returns.  An integer or bool result
// gets 0.
inline ::mlx::core::array out_of_range_fill(::mlx::core::Dtype dt) {
    namespace mx = ::mlx::core;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (dt == mx::complex64)
        return mx::array(mx::complex64_t{nan, nan});
    if (mx::issubdtype(dt, mx::floating))
        return mx::array(nan, dt);
    return mx::array(0, dt);
}

// ``gathered`` with every out-of-range position replaced by the fill.
// ``in_range`` broadcasts against it.
inline ::mlx::core::array fill_out_of_range(const ::mlx::core::array& gathered,
                                            const ::mlx::core::array& in_range) {
    return ::mlx::core::where(in_range, gathered, out_of_range_fill(gathered.dtype()));
}

// ``take_along_axis`` under policy B.
inline ::mlx::core::array gpu_take_along_axis(const ::mlx::core::array& src,
                                              const ::mlx::core::array& idx,
                                              int axis,
                                              NegativeIndex negative = NegativeIndex::Wrap) {
    namespace mx = ::mlx::core;
    const int ax = axis < 0 ? axis + static_cast<int>(src.ndim()) : axis;
    const auto extent = static_cast<std::int64_t>(src.shape(ax));
    if (extent == 0) {
        auto shape = src.shape();
        shape[static_cast<std::size_t>(ax)] = idx.shape(ax);
        return mx::full(mx::broadcast_shapes(shape, idx.shape()), out_of_range_fill(src.dtype()));
    }
    auto ix = gpu_axis_index(idx, extent, negative);
    return fill_out_of_range(mx::take_along_axis(src, ix.safe, ax), ix.in_range);
}

// ``take(src, idx, axis)`` under policy B: whole slices of ``src`` picked
// along ``axis``.  The result is ``src.shape[:axis] + idx.shape +
// src.shape[axis + 1:]``.
inline ::mlx::core::array gpu_take(const ::mlx::core::array& src,
                                   const ::mlx::core::array& idx,
                                   int axis,
                                   NegativeIndex negative) {
    namespace mx = ::mlx::core;
    const int ndim = static_cast<int>(src.ndim());
    const int ax = axis < 0 ? axis + ndim : axis;
    mx::Shape out_shape(src.shape().begin(), src.shape().begin() + ax);
    mx::Shape mask_shape(static_cast<std::size_t>(ax), 1);
    for (auto s : idx.shape()) {
        out_shape.push_back(s);
        mask_shape.push_back(s);
    }
    for (int d = ax + 1; d < ndim; ++d) {
        out_shape.push_back(src.shape(d));
        mask_shape.push_back(1);
    }
    const auto extent = static_cast<std::int64_t>(src.shape(ax));
    if (extent == 0)
        return mx::full(out_shape, out_of_range_fill(src.dtype()));
    auto ix = gpu_axis_index(idx, extent, negative);
    return fill_out_of_range(mx::take(src, ix.safe, ax), mx::reshape(ix.in_range, mask_shape));
}

// The reductions a scatter can apply.
enum class ScatterReduce { Add, Max, Min, Prod };

// The update that leaves a position as it was, which is what an out-of-range
// update becomes before it is scattered.  For a sum that is -0.0 and not
// +0.0: ``x + -0.0`` is ``x`` for every ``x`` including -0.0, while
// ``-0.0 + 0.0`` is +0.0.  MLX's atomic max and min write only when the
// update beats the value held, which -inf and +inf never do, NaN included.
// Its atomic product writes ``held * 1``, the same bits.
inline ::mlx::core::array scatter_identity(ScatterReduce op, ::mlx::core::Dtype dt) {
    namespace mx = ::mlx::core;
    const bool complex = dt == mx::complex64;
    const bool floating = mx::issubdtype(dt, mx::floating);
    if (op == ScatterReduce::Add) {
        if (complex)
            return mx::array(mx::complex64_t{-0.0f, -0.0f});
        return floating ? mx::array(-0.0f, dt) : mx::array(0, dt);
    }
    if (op == ScatterReduce::Prod)
        return mx::array(1, dt);
    if (complex)
        ErrorBuilder("gpu_backend::scatter_reduce")
            .not_implemented("amax / amin have no order on complex values");
    const bool lowest = op == ScatterReduce::Max;
    if (floating) {
        const float inf = std::numeric_limits<float>::infinity();
        return mx::array(lowest ? -inf : inf, dt);
    }
    if (dt == mx::bool_)
        return mx::array(!lowest);
    auto bound = [&](auto tag) {
        using T = decltype(tag);
        return mx::array(lowest ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max(),
                         dt);
    };
    if (dt == mx::int8)
        return bound(std::int8_t{});
    if (dt == mx::int16)
        return bound(std::int16_t{});
    if (dt == mx::int32)
        return bound(std::int32_t{});
    if (dt == mx::int64)
        return bound(std::int64_t{});
    if (dt == mx::uint8)
        return bound(std::uint8_t{});
    if (dt == mx::uint16)
        return bound(std::uint16_t{});
    if (dt == mx::uint32)
        return bound(std::uint32_t{});
    if (dt == mx::uint64)
        return bound(std::uint64_t{});
    ErrorBuilder("gpu_backend::scatter_reduce").not_implemented("dtype not supported");
}

// An axis scatter of ``op`` under policy B.  Each out-of-range update
// becomes the identity of ``op`` and is aimed at an in-range position,
// which it leaves as it was.  ``scatter(b, i, v)`` performs the MLX scatter
// on ``scatter_on_index_corner``'s operands.
template <class Scatter>
::mlx::core::array gpu_scatter_reduce_axis(const ::mlx::core::array& base,
                                           const ::mlx::core::array& idx,
                                           const ::mlx::core::array& src,
                                           int dim,
                                           ScatterReduce op,
                                           Scatter&& scatter) {
    const int d = dim < 0 ? dim + static_cast<int>(base.ndim()) : dim;
    if (base.shape(d) == 0)
        return base;
    auto ix = gpu_axis_index(idx, base.shape(d));
    const auto identity = scatter_identity(op, base.dtype());
    return scatter_on_index_corner(
        base, ix.safe, src, d,
        [&](const ::mlx::core::array& b, const ::mlx::core::array& i, const ::mlx::core::array& v) {
            return scatter(b, i, ::mlx::core::where(ix.in_range, v, identity));
        });
}

// Where an overwrite scatter sends each update under policy B: the index
// itself where it is in range, and ``extent`` otherwise.  ``extent`` names a
// one-wide sink slab that the caller appends past the end of the axis and
// cuts off afterwards.  Unlike a sum, a write cannot be made harmless: any
// value aimed at a real position overwrites it.  So an out-of-range write
// gets a position of its own.
inline ::mlx::core::array gpu_sink_index(const ::mlx::core::array& idx, std::int64_t extent) {
    auto ix = gpu_axis_index(idx, extent);
    return ::mlx::core::where(ix.in_range, ix.safe, ::mlx::core::array(extent, ix.safe.dtype()));
}

// ``base`` with the one-wide sink slab appended along ``dim``.
inline ::mlx::core::array with_sink_slab(const ::mlx::core::array& base, int dim) {
    auto slab = base.shape();
    slab[static_cast<std::size_t>(dim)] = 1;
    return ::mlx::core::concatenate({base, ::mlx::core::zeros(slab, base.dtype())}, dim);
}

// ``padded`` with the sink slab cut off again, which leaves ``shape``.
inline ::mlx::core::array without_sink_slab(const ::mlx::core::array& padded,
                                            const ::mlx::core::Shape& shape) {
    return ::mlx::core::contiguous(
        ::mlx::core::slice(padded, ::mlx::core::Shape(shape.size(), 0), shape));
}

}  // namespace backend
}  // namespace lucid
