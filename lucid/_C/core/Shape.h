// lucid/_C/core/Shape.h
//
// Type aliases and utility functions describing a tensor's geometric layout:
// the dimension vector :type:`Shape` and the byte-stride vector :type:`Stride`.
//
// Lucid follows the row-major (C-contiguous) convention used by NumPy and the
// reference framework — innermost dimension varies fastest in memory, and
// :func:`contiguous_stride` produces the canonical stride for that layout.
// Strides are stored as **byte** offsets (not element counts), so kernels can
// move through memory without a dtype lookup at every step.
//
// See Also
// --------
// :class:`Dtype`      — element size used by :func:`contiguous_stride`.
// :class:`TensorMeta` — owns the per-tensor :type:`Shape` and :type:`Stride`.

#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lucid {

// Sequence of dimension sizes describing a tensor's shape.
//
// Dimensions are stored as signed ``int64`` so the same vector can carry
// negative sentinels (typically ``-1``) for unknown / dynamic extents — for
// example before a deferred-shape op has been materialised.  Free-form
// construction via braced-init is supported and idiomatic:
//
// Examples
// --------
// ::
//
//     Shape s{B, C, H, W};                    // 4-D image batch
//     Shape scalar_shape;                     // empty vector → scalar (numel=1)
//     Shape dynamic{-1, 256};                 // first axis unresolved
//
// See Also
// --------
// :func:`shape_numel`        — total element count.
// :func:`contiguous_stride`  — canonical row-major byte stride.
using Shape = std::vector<std::int64_t>;

// Sequence of per-axis byte strides.
//
// ``stride[i]`` is the **number of bytes** to advance in the underlying
// :class:`Storage` buffer to move one element along axis ``i``.  Storing the
// stride in bytes rather than element counts lets the same vector type cover
// every dtype without per-call multiplications inside hot loops.
//
// A tensor is *contiguous* when its stride vector equals
// :func:`contiguous_stride` for its shape and element size.
using Stride = std::vector<std::int64_t>;

// Returns the total number of elements implied by a shape.
//
// The product is computed in :type:`std::size_t` to avoid sign issues when
// the caller wants to use the result for allocation sizing.
//
// Parameters
// ----------
// shape : Shape
//     Shape vector to reduce.  May be empty (scalar) or contain negative
//     sentinel values.
//
// Returns
// -------
// std::size_t
//     Product of all dimensions.  ``1`` for an empty shape (scalar
//     convention); ``0`` as soon as any dimension is negative — the caller
//     is responsible for not allocating storage in that case.
//
// Notes
// -----
// The early-return-zero behaviour for negative dims is deliberate: it lets
// dynamic-shape pipelines pass an unresolved :type:`Shape` through generic
// code paths without manually filtering for sentinels.
//
// Examples
// --------
// ::
//
//     shape_numel({})            // 1   (scalar)
//     shape_numel({3, 4})        // 12
//     shape_numel({-1, 8})       // 0   (unresolved leading dim)
inline std::size_t shape_numel(const Shape& shape) {
    if (shape.empty())
        return 1;
    std::size_t n = 1;
    for (auto d : shape) {
        if (d < 0)
            return 0;
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

// Computes the row-major (C-contiguous) byte-stride vector for a tensor with
// the given shape and element size.
//
// The stride is built back-to-front: the innermost (last) axis has stride
// ``elem_size``, and each outer axis's stride is the product of the inner
// stride and the inner dimension size.  This matches NumPy's default
// ``'C'`` order and is the layout expected by Apple Accelerate BLAS / vDSP
// routines.
//
// Parameters
// ----------
// shape : Shape
//     Target shape.  Empty shape yields an empty stride.
// elem_size : std::size_t
//     Element size in bytes — typically obtained from :func:`dtype_size`.
//
// Returns
// -------
// Stride
//     Vector of the same length as ``shape``.  When the result is fed back
//     into :class:`TensorMeta`, :func:`TensorImpl::is_contiguous` returns
//     ``true``.
//
// Math
// ----
// $$\text{stride}[i] = \text{elem\_size} \cdot \prod_{j > i} \text{shape}[j]$$
//
// Examples
// --------
// ::
//
//     contiguous_stride({2, 3, 4}, 4)    // {48, 16, 4}  (float32)
//     contiguous_stride({}, 4)           // {}           (scalar)
//
// See Also
// --------
// :func:`TensorMeta::is_contiguous_for`
//     Reverse check — verifies a stored stride against the canonical one.
inline Stride contiguous_stride(const Shape& shape, std::size_t elem_size) {
    Stride s(shape.size());
    if (shape.empty())
        return s;
    std::int64_t acc = static_cast<std::int64_t>(elem_size);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
        s[static_cast<std::size_t>(i)] = acc;
        acc *= shape[static_cast<std::size_t>(i)];
    }
    return s;
}

}  // namespace lucid
