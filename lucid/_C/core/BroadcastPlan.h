// lucid/_C/core/BroadcastPlan.h
//
// A broadcast, described as runs of memory rather than elements.
//
// Pure index arithmetic, shared by every CPU path that materialises a
// broadcast or sums one back: the backend's ``broadcast`` and
// ``reduce_broadcast``, and the binary-op kernels' ``broadcast_cpu``.  They
// each walked the output one element at a time, recomputing an N-d
// coordinate for every element.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "Shape.h"

namespace lucid {

// How a broadcast from ``src_shape`` to ``dst_shape`` lays out in runs.
//
// The output is ``outer`` blocks, one after another.  Block ``o`` reads the
// ``inner`` contiguous source elements that start at the offset its outer
// coordinate selects, and writes each of them ``rep`` times in a row — so a
// scalar is one block of one element repeated everywhere, a row vector is
// one contiguous copy per row, and a column is one fill per row.  Walking
// runs instead of elements is what lets a copy or a fill do the work: the
// per-element coordinate walk this replaced cost 2.3 ns an element, fifty
// times a same-shape multiply, and every bias add and scalar product paid
// it.
struct BroadcastPlan {
    std::size_t outer = 1;
    std::size_t inner = 1;
    std::size_t rep = 1;
    std::vector<std::size_t> outer_dims;    // dst's leading dims, [0, k)
    std::vector<std::size_t> outer_stride;  // the source's stride per such dim; 0 = broadcast
};

// Build the :class:`BroadcastPlan` for ``src_shape`` -> ``dst_shape``.
//
// Parameters
// ----------
// src_shape : const Shape&
//     Broadcastable to ``dst_shape`` (right-aligned, size-1 dims stretch).
// dst_shape : const Shape&
//     The full shape.
//
// Returns
// -------
// BroadcastPlan
//     ``outer * inner * rep`` equals the element count of ``dst_shape``.
inline BroadcastPlan plan_broadcast(const Shape& src_shape, const Shape& dst_shape) {
    const std::size_t nd = dst_shape.size();
    std::vector<std::size_t> dst(nd), padded(nd, 1);
    for (std::size_t d = 0; d < nd; ++d)
        dst[d] = static_cast<std::size_t>(dst_shape[d]);
    for (std::size_t i = 0; i < src_shape.size(); ++i)
        padded[nd - src_shape.size() + i] = static_cast<std::size_t>(src_shape[i]);

    BroadcastPlan plan;
    // Trailing dims the source holds one of: each source element repeats.
    std::size_t j = nd;
    while (j > 0 && padded[j - 1] == 1)
        --j;
    for (std::size_t d = j; d < nd; ++d)
        plan.rep *= dst[d];
    // The contiguous source run: dims before those that it holds in full.
    std::size_t k = j;
    while (k > 0 && padded[k - 1] == dst[k - 1])
        --k;
    for (std::size_t d = k; d < j; ++d)
        plan.inner *= dst[d];

    std::vector<std::size_t> stride(nd, 0);
    std::size_t step = 1;
    for (std::size_t d = nd; d-- > 0;) {
        stride[d] = padded[d] == 1 ? 0 : step;
        step *= padded[d];
    }
    for (std::size_t d = 0; d < k; ++d) {
        plan.outer *= dst[d];
        plan.outer_dims.push_back(dst[d]);
        plan.outer_stride.push_back(stride[d]);
    }
    return plan;
}

// Call ``fn(o, src_offset)`` for every block of ``plan``, in order.
template <class F>
void for_each_broadcast_block(const BroadcastPlan& plan, F&& fn) {
    const std::size_t nd = plan.outer_dims.size();
    std::vector<std::size_t> coord(nd, 0);
    std::size_t offset = 0;
    for (std::size_t o = 0; o < plan.outer; ++o) {
        fn(o, offset);
        for (std::size_t d = nd; d-- > 0;) {
            offset += plan.outer_stride[d];
            if (++coord[d] < plan.outer_dims[d])
                break;
            offset -= plan.outer_stride[d] * coord[d];
            coord[d] = 0;
        }
    }
}

// Write the broadcast of ``src`` (shaped as the plan's source) into ``dst``.
//
// Parameters
// ----------
// plan : const BroadcastPlan&
//     From :func:`plan_broadcast`.
// src : const T*
//     The source elements, contiguous.
// dst : T*
//     Room for ``outer * inner * rep`` elements.
template <class T>
void broadcast_runs(const BroadcastPlan& plan, const T* src, T* dst) {
    const std::size_t block = plan.inner * plan.rep;
    if (block == 0)
        return;
    for_each_broadcast_block(plan, [&](std::size_t o, std::size_t offset) {
        const T* from = src + offset;
        T* to = dst + o * block;
        if (plan.rep == 1) {
            std::copy(from, from + plan.inner, to);
        } else {
            for (std::size_t i = 0; i < plan.inner; ++i)
                std::fill_n(to + i * plan.rep, plan.rep, from[i]);
        }
    });
}

// Add a full-shape ``grad`` back onto ``dst``, shaped as the plan's source.
//
// Parameters
// ----------
// plan : const BroadcastPlan&
//     From :func:`plan_broadcast`.
// grad : const T*
//     ``outer * inner * rep`` elements, contiguous.
// dst : T*
//     The source-shaped accumulator; added to, not overwritten.
//
// Notes
// -----
// Every destination receives its contributions in increasing output order —
// the order an element-by-element walk uses — so a float sum is the same
// bits it always was; a vectorised or pairwise sum would not be.
template <class T>
void sum_broadcast_runs(const BroadcastPlan& plan, const T* grad, T* dst) {
    const std::size_t block = plan.inner * plan.rep;
    if (block == 0)
        return;
    for_each_broadcast_block(plan, [&](std::size_t o, std::size_t offset) {
        const T* g = grad + o * block;
        T* to = dst + offset;
        if (plan.rep == 1) {
            for (std::size_t i = 0; i < plan.inner; ++i)
                to[i] += g[i];
        } else {
            for (std::size_t i = 0; i < plan.inner; ++i) {
                T acc = to[i];
                const T* run = g + i * plan.rep;
                for (std::size_t r = 0; r < plan.rep; ++r)
                    acc += run[r];
                to[i] = acc;
            }
        }
    });
}

}  // namespace lucid
