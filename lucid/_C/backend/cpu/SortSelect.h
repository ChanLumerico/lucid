// lucid/_C/backend/cpu/SortSelect.h
//
// The CPU sort / top-k selection kernel and the scatter-add that is its
// adjoint (sort, argsort, topk and their backward, as_strided backward).
//
// The positions they produce and consume are int64, the index dtype every
// index-returning op answers with (argmax, nonzero, the reference's sort /
// argsort / topk).  The scatter still reads an int32 index too, because
// as_strided's backward builds one.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::backend::cpu {

// ``outer`` / ``inner`` extents around ``axis`` of ``shape``.
inline std::pair<std::size_t, std::size_t> around_axis(const Shape& shape, int axis) {
    std::size_t outer = 1;
    std::size_t inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    return {outer, inner};
}

// The first ``output_shape[axis]`` elements of each line along ``axis`` in
// sorted order, and the int64 position each came from.  Equal keys keep
// their input order, in a full sort and a top-k alike — what sort / argsort
// document, and what the GPU path does.
template <typename T>
std::pair<CpuStorage, CpuStorage> sort_select_cpu(const CpuStorage& input,
                                                  const Shape& input_shape,
                                                  const Shape& output_shape,
                                                  int axis,
                                                  Dtype dt,
                                                  bool descending) {
    const std::size_t n_out = shape_numel(output_shape);
    CpuStorage values{allocate_aligned_bytes(n_out * dtype_size(dt), Device::CPU),
                      n_out * dtype_size(dt), dt};
    CpuStorage indices{allocate_aligned_bytes(n_out * sizeof(std::int64_t), Device::CPU),
                       n_out * sizeof(std::int64_t), Dtype::I64};

    const auto* src = reinterpret_cast<const T*>(input.ptr.get());
    auto* dst = reinterpret_cast<T*>(values.ptr.get());
    auto* idx_dst = reinterpret_cast<std::int64_t*>(indices.ptr.get());
    const auto [outer, inner] = around_axis(input_shape, axis);
    const std::size_t L = static_cast<std::size_t>(input_shape[static_cast<std::size_t>(axis)]);
    const std::size_t K = static_cast<std::size_t>(output_shape[static_cast<std::size_t>(axis)]);

    std::vector<std::int64_t> order(L);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            std::iota(order.begin(), order.end(), std::int64_t{0});
            // NaN sorts as the largest value, so ascending puts it last and
            // descending puts it first — the reference framework's convention.
            //
            // This is not only about where the NaN lands.  ``lv < rv`` is
            // false in *both* directions when either side is NaN, so the
            // comparator is not a strict weak ordering and ``std::sort`` has
            // undefined behaviour: ``[1, nan, 3, -1]`` came back
            // ``[1, nan, -1, 3]``, and a longer array can walk the pivot off
            // the end of the buffer.  Ordering NaN makes the comparator valid.
            //
            // Equal values are ordered by position, lowest first.  A full
            // sort is stable anyway, but ``partial_sort`` (a top-k) is not:
            // ``topk([0, 5, 4, 4], 3)`` named 1, 3, 2.
            auto cmp = [&](std::int64_t lhs, std::int64_t rhs) {
                const T lv = src[(o * L + static_cast<std::size_t>(lhs)) * inner + j];
                const T rv = src[(o * L + static_cast<std::size_t>(rhs)) * inner + j];
                if constexpr (std::is_floating_point_v<T>) {
                    const bool ln = std::isnan(lv);
                    const bool rn = std::isnan(rv);
                    if (ln || rn) {
                        if (ln && rn)
                            return lhs < rhs;
                        return descending ? ln : rn;
                    }
                }
                if (lv == rv)
                    return lhs < rhs;
                return descending ? (lv > rv) : (lv < rv);
            };
            if (K == L)
                std::stable_sort(order.begin(), order.end(), cmp);
            else
                std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(K),
                                  order.end(), cmp);
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t out_flat = (o * K + k) * inner + j;
                dst[out_flat] = src[(o * L + static_cast<std::size_t>(order[k])) * inner + j];
                idx_dst[out_flat] = order[k];
            }
        }
    }
    return {std::move(values), std::move(indices)};
}

// ``dst[..., idx[..., k, ...], ...] += grad[..., k, ...]`` along ``axis``, into
// ``dst`` of ``output_shape`` (zeroed by the caller).  A negative position
// counts from the end of the axis.
template <typename T, typename I>
void scatter_add_axis_lines(T* dst,
                            const T* grad,
                            const I* idx,
                            const Shape& output_shape,
                            const Shape& grad_shape,
                            int axis) {
    const auto [outer, inner] = around_axis(output_shape, axis);
    const auto L = static_cast<std::int64_t>(output_shape[static_cast<std::size_t>(axis)]);
    const std::size_t K = static_cast<std::size_t>(grad_shape[static_cast<std::size_t>(axis)]);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t grad_flat = (o * K + k) * inner + j;
                std::int64_t src_k = static_cast<std::int64_t>(idx[grad_flat]);
                if (src_k < 0)
                    src_k += L;
                dst[(o * static_cast<std::size_t>(L) + static_cast<std::size_t>(src_k)) * inner +
                    j] += grad[grad_flat];
            }
        }
    }
}

// :func:`scatter_add_axis_lines` on the index's own width.
template <typename T>
void scatter_add_axis_any_index(T* dst,
                                const T* grad,
                                const CpuStorage& idx,
                                const Shape& output_shape,
                                const Shape& grad_shape,
                                int axis) {
    if (idx.dtype == Dtype::I64)
        scatter_add_axis_lines(dst, grad, reinterpret_cast<const std::int64_t*>(idx.ptr.get()),
                               output_shape, grad_shape, axis);
    else
        scatter_add_axis_lines(dst, grad, reinterpret_cast<const std::int32_t*>(idx.ptr.get()),
                               output_shape, grad_shape, axis);
}

}  // namespace lucid::backend::cpu
