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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lucid {

// Sequence of dimension sizes describing a tensor's shape.
//
// Dimensions are signed ``int64`` so that an op's *argument* can carry a
// wildcard — reshape's inferred ``-1``, expand's keep-this-size ``-1`` — until
// the op resolves it.  A tensor's shape is always concrete: every extent is
// ``>= 0``, and :func:`shape_numel` / :func:`contiguous_stride` refuse any
// other (the tensor constructor builds its stride through the latter).
// Free-form construction via braced-init is supported and idiomatic:
//
// Examples
// --------
// ::
//
//     Shape s{B, C, H, W};                    // 4-D image batch
//     Shape scalar_shape;                     // empty vector → scalar (numel=1)
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

// Refuses a shape with a negative extent.
//
// Raises :class:`InvalidArgument` naming the innermost op on the error-context
// stack and the first negative extent, in the reference framework's words:
// ``zeros: Trying to create tensor with negative dimension -3: [2, -3]``.
// Defined out of line, in ``ErrorBuilder.cpp``: the op name lives on
// :class:`ErrorContext`, whose header includes this one.  Not exported: only
// the engine itself calls it.
[[noreturn]] void throw_negative_dim(const Shape& shape);

// Refuses ``shape`` unless every extent is ``>= 0``.
//
// For an op that must turn a shape away before it dispatches: the backend
// sizes its work by the shape it is handed, before any tensor is built.
inline void check_concrete_shape(const Shape& shape) {
    for (auto d : shape)
        if (d < 0)
            throw_negative_dim(shape);
}

// Returns the total number of elements implied by a shape — the one
// element-count rule; :func:`TensorMeta::numel` and every allocation sized
// by a shape go through it.
//
// The product is computed in :type:`std::size_t` so the result can size an
// allocation directly.
//
// Parameters
// ----------
// shape : Shape
//     A concrete shape.  May be empty (scalar).
//
// Returns
// -------
// std::size_t
//     Product of all dimensions.  ``1`` for an empty shape (scalar
//     convention).
//
// Raises
// ------
// InvalidArgument
//     A dimension is negative (:func:`throw_negative_dim`).  It has no
//     count: as a ``size_t`` it is ``2^64 - 1``, and a kernel that walks
//     that count runs off its buffer.
//
// Examples
// --------
// ::
//
//     shape_numel({})            // 1   (scalar)
//     shape_numel({3, 4})        // 12
//     shape_numel({-1, 8})       // throws InvalidArgument
inline std::size_t shape_numel(const Shape& shape) {
    std::size_t n = 1;
    for (auto d : shape) {
        if (d < 0)
            throw_negative_dim(shape);
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
//     A concrete shape.  Empty shape yields an empty stride.
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
// Raises
// ------
// InvalidArgument
//     A dimension is negative (:func:`throw_negative_dim`).  The tensor
//     constructor derives its stride here, so no tensor is built with one.
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
        const std::int64_t d = shape[static_cast<std::size_t>(i)];
        if (d < 0)
            throw_negative_dim(shape);
        s[static_cast<std::size_t>(i)] = acc;
        acc *= std::max<std::int64_t>(d, 1);
    }
    return s;
}

}  // namespace lucid
