// lucid/_C/core/GpuStrided.h
//
// Reading and writing one member of a Metal view family through its own
// geometry — the GPU counterpart of StridedCopy.h.
//
// An MLX array is a value, so a Metal view cannot share bytes with its base
// the way a CPU view does.  A family instead keeps one current value, the
// root (ViewFamily::root), and every member is a geometry over it: a shape,
// byte strides and a byte offset counted in the root's row-major elements,
// exactly as a CPU view counts them in its buffer.  A read builds the
// member's elements from the root; a write builds the next root with the
// member's elements replaced.  Both are lazy MLX graph nodes.  Strides are
// in bytes and non-negative, as make_view's are.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <mlx/ops.h>

#include "Shape.h"

namespace lucid::gpu_strided {

// The geometry of one member, in elements of the root.
struct Geometry {
    Shape shape;
    std::vector<std::int64_t> stride;  // elements
    std::int64_t offset = 0;           // elements
    std::int64_t numel = 0;
    bool contiguous = false;

    Geometry(const Shape& shp, const Stride& byte_stride, std::size_t byte_offset, std::size_t elem)
        : shape(shp),
          offset(static_cast<std::int64_t>(byte_offset / elem)),
          numel(static_cast<std::int64_t>(shape_numel(shp))) {
        stride.reserve(byte_stride.size());
        for (const auto s : byte_stride)
            stride.push_back(s / static_cast<std::int64_t>(elem));
        contiguous = byte_stride == contiguous_stride(shp, elem);
    }

    // Whether the member reads every element of a root of ``root_numel``
    // elements, in the root's order.
    bool covers(std::int64_t root_numel) const {
        return contiguous && offset == 0 && numel == root_numel;
    }
};

inline ::mlx::core::Shape to_mlx(const Shape& shape) {
    return ::mlx::core::Shape(shape.begin(), shape.end());
}

// The member's elements, as a row-contiguous array of its shape.
inline ::mlx::core::array read(const ::mlx::core::array& root, const Geometry& g) {
    namespace mx = ::mlx::core;
    const auto root_numel = static_cast<std::int64_t>(root.size());
    if (g.numel == 0)
        return mx::zeros(to_mlx(g.shape), root.dtype());
    if (g.covers(root_numel))
        return mx::reshape(root, to_mlx(g.shape));
    if (g.contiguous) {
        const auto lo = static_cast<int>(g.offset);
        const auto hi = static_cast<int>(g.offset + g.numel);
        return mx::reshape(mx::slice(mx::flatten(root), {lo}, {hi}), to_mlx(g.shape));
    }
    // as_strided reads a flattened (row-contiguous) copy of its input, and
    // its result is a strided window onto that; laid out again so code that
    // reads the buffer directly sees the member's own order.
    mx::Strides strides(g.stride.begin(), g.stride.end());
    return mx::contiguous(
        mx::as_strided(root, to_mlx(g.shape), std::move(strides), static_cast<size_t>(g.offset)));
}

// The member as a strided slice of the root reshaped to ``dims``: the plan
// slice_update needs, or false when the geometry is not one.
//
// The member's axes longer than one, ordered by stride, outermost first,
// become the slice's axes.  Every axis but the innermost steps by one over
// a root axis as long as the next axis's stride; the innermost steps by its
// own stride over a root axis of unit stride.  That works when each stride
// divides the one before it, the root's size divides by the outermost, and
// every axis's run stays inside its root axis.
struct SlicePlan {
    ::mlx::core::Shape dims, start, stop, step;
    std::vector<int> order;  // member axes (longer than one), outermost first
};

// The member's axes longer than one, outermost (largest stride) first.
inline std::vector<int> axes_by_stride(const Geometry& g) {
    std::vector<int> axes;
    for (std::size_t d = 0; d < g.shape.size(); ++d)
        if (g.shape[d] > 1)
            axes.push_back(static_cast<int>(d));
    std::stable_sort(axes.begin(), axes.end(),
                     [&](int a, int b) { return g.stride[a] > g.stride[b]; });
    return axes;
}

// The plan's start and stop on each root axis, of ``unit`` elements per step
// and ``dims`` long: false when an axis's run leaves its root axis.
inline bool fill_plan(const Geometry& g,
                      const std::vector<int>& axes,
                      const std::vector<std::int64_t>& unit,
                      const std::vector<std::int64_t>& dims,
                      const std::vector<std::int64_t>& step,
                      SlicePlan& plan) {
    constexpr auto kMax = static_cast<std::int64_t>(std::numeric_limits<int>::max());
    plan = SlicePlan{};
    plan.order = axes;
    std::int64_t rest = g.offset;
    for (std::size_t k = 0; k < axes.size(); ++k) {
        const std::int64_t start = rest / unit[k];
        rest -= start * unit[k];
        const std::int64_t last = start + (g.shape[axes[k]] - 1) * step[k];
        if (last >= dims[k] || dims[k] > kMax)
            return false;
        plan.dims.push_back(static_cast<int>(dims[k]));
        plan.start.push_back(static_cast<int>(start));
        plan.stop.push_back(static_cast<int>(last + 1));
        plan.step.push_back(static_cast<int>(step[k]));
    }
    return true;
}

inline bool plan_slice(const Geometry& g, std::int64_t root_numel, SlicePlan& plan) {
    const std::vector<int> axes = axes_by_stride(g);
    const std::size_t m = axes.size();
    if (m == 0)
        return false;
    // Root axis k steps ``unit[k]`` elements: the member's own stride on
    // every axis but the innermost, which walks a root axis of unit stride
    // by its own stride.
    std::vector<std::int64_t> unit(m, 1), dims(m), step(m, 1);
    for (std::size_t k = 0; k + 1 < m; ++k)
        unit[k] = g.stride[axes[k]];
    step[m - 1] = g.stride[axes[m - 1]];
    if (step[m - 1] <= 0)
        return false;
    for (std::size_t k = 1; k < m; ++k) {
        if (unit[k - 1] <= 0 || unit[k - 1] % unit[k] != 0)
            return false;
        dims[k] = unit[k - 1] / unit[k];
    }
    if (root_numel % unit[0] != 0)
        return false;
    dims[0] = root_numel / unit[0];
    return fill_plan(g, axes, unit, dims, step, plan);
}

// Each of the member's elements' position in the root, in the member's
// row-major order, as a flat index array.
inline ::mlx::core::array positions(const Geometry& g, std::int64_t root_numel) {
    namespace mx = ::mlx::core;
    const auto dt = root_numel > std::numeric_limits<std::int32_t>::max() ? mx::int64 : mx::int32;
    mx::array at(g.offset, dt);
    const std::size_t nd = g.shape.size();
    for (std::size_t d = 0; d < nd; ++d) {
        if (g.shape[d] <= 1)
            continue;
        mx::Shape axis_shape(nd, 1);
        axis_shape[d] = static_cast<int>(g.shape[d]);
        const mx::array steps = mx::multiply(
            mx::arange(0.0, static_cast<double>(g.shape[d]), 1.0, dt), mx::array(g.stride[d], dt));
        at = mx::add(at, mx::reshape(steps, axis_shape));
    }
    return mx::flatten(mx::broadcast_to(at, to_mlx(g.shape)));
}

// The next root: ``root`` with the member's elements replaced by ``value``
// (the member's shape, the root's dtype).  The root keeps its shape.  The
// member must not repeat an element (the caller refuses one that does).
inline ::mlx::core::array
write(const ::mlx::core::array& root, const ::mlx::core::array& value, const Geometry& g) {
    namespace mx = ::mlx::core;
    const auto root_numel = static_cast<std::int64_t>(root.size());
    const mx::Shape root_shape = root.shape();
    if (g.numel == 0)
        return root;
    if (g.covers(root_numel))
        return mx::reshape(value, root_shape);
    const mx::array flat = mx::flatten(root);
    const mx::array run = mx::reshape(value, {static_cast<int>(g.numel)});
    if (g.contiguous || g.numel == 1) {
        const auto lo = static_cast<int>(g.offset);
        return mx::reshape(mx::slice_update(flat, run, {lo}, {lo + static_cast<int>(g.numel)}),
                           root_shape);
    }
    if (SlicePlan plan; plan_slice(g, root_numel, plan)) {
        // The value with its size-1 axes dropped and its axes put in the
        // plan's order, outermost first.
        mx::Shape kept;
        std::vector<int> rank(g.shape.size(), -1);
        for (std::size_t d = 0; d < g.shape.size(); ++d)
            if (g.shape[d] > 1) {
                rank[d] = static_cast<int>(kept.size());
                kept.push_back(static_cast<int>(g.shape[d]));
            }
        std::vector<int> perm;
        perm.reserve(plan.order.size());
        for (const int axis : plan.order)
            perm.push_back(rank[static_cast<std::size_t>(axis)]);
        const mx::array update = mx::transpose(mx::reshape(value, kept), perm);
        return mx::reshape(mx::slice_update(mx::reshape(root, plan.dims), update, plan.start,
                                            plan.stop, plan.step),
                           root_shape);
    }
    return mx::reshape(mx::put_along_axis(flat, positions(g, root_numel), run, 0), root_shape);
}

}  // namespace lucid::gpu_strided
