// lucid/_C/core/Shape.h
//
// Type aliases and utility functions for tensor shapes and strides.
// Shape is a signed 64-bit dimension vector to accommodate negative sentinel
// values (e.g. a dim of -1 representing an unknown / dynamic extent).
// Stride stores byte offsets between successive elements along each axis,
// following the same row-major convention as NumPy and PyTorch.

#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lucid {

// Sequence of dimension sizes.  Negative entries are treated as zero elements
// by shape_numel, which is useful for representing tensors whose size is not
// yet materialised.
using Shape = std::vector<std::int64_t>;

// Sequence of byte strides, one per dimension.  stride[i] is the number of
// bytes to advance in the underlying buffer to move one step along axis i.
using Stride = std::vector<std::int64_t>;

// Returns the total number of elements implied by shape.
//
// An empty shape (scalar) returns 1.  Any negative dimension causes an early
// return of 0 — the caller is responsible for not allocating storage in that
// case.
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

// Computes the row-major (C-contiguous) byte stride vector for a tensor with
// the given shape and element size.
//
// The stride is computed back-to-front: the innermost (last) dimension has
// stride elem_size, and each outer dimension's stride is the product of the
// inner stride and the inner dimension size.  This matches the NumPy default
// ('C' order) and is the layout expected by Accelerate BLAS routines.
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
