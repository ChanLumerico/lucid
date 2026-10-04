// lucid/_C/backend/cpu/Shape.h
//
// CPU shape-transformation helper: permute_copy performs an N-D transpose by
// copying elements in the permuted order into a fresh densely-packed buffer.
// This is used by CpuBackend::permute(), which on the CPU runs only for the
// backward of permute / transpose / swapaxes / mT (the forward is a view).
//
// The permutation perm[d] specifies which input axis maps to output axis d,
// following NumPy conventions (e.g. perm = {2, 0, 1} maps (H, W, C) → (C, H, W)).
// Output strides are computed from the output shape in C (row-major) order.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../api.h"

namespace lucid::backend::cpu {

// Copies a single-precision tensor into a permuted, densely-packed output
// buffer using the supplied axis permutation.
//
// The output is the input viewed through its C-order strides reordered by
// ``perm``, packed by :func:`strided::pack` — axes merged, contiguous runs
// moved by memcpy, a merged 2-D transpose copied in tiles.  This is an
// out-of-place layout transform — there is no in-place fast path.
//
// Parameters
// ----------
// in : const float*
//     Source buffer laid out densely (C-order) according to ``in_shape``.
// out : float*
//     Destination buffer.  Caller must pre-allocate
//     ``numel(in_shape) * sizeof(float)`` bytes.  May not alias ``in``.
// in_shape : const std::vector<int64_t>&
//     Shape of the source tensor.  An empty vector is treated as a 0-D scalar
//     and produces a single-element copy.
// perm : const std::vector<int>&
//     Axis permutation of length ``in_shape.size()``.  ``perm[d]`` names the
//     source axis whose extent becomes output axis ``d``.  Must be a valid
//     permutation of ``[0, ndim)``.
//
// Shape
// -----
// ``out_shape[d] == in_shape[perm[d]]`` for every ``d``.
//
// Examples
// --------
// ``perm = {2, 0, 1}`` over ``in_shape = {H, W, C}`` yields a CHW output.
// ``perm = {0, 2, 3, 1}`` over ``in_shape = {N, C, H, W}`` performs an NHWC
// re-layout.
//
// Notes
// -----
// Cost is one pass over the bytes plus one odometer step per run, not per
// element; a permutation that keeps the last axis in place moves whole rows.
// Single-threaded.
//
// See Also
// --------
// permute_copy_f64, permute_copy_i32, permute_copy_i64 : Same operation for
//     other element types.
LUCID_INTERNAL void permute_copy_f32(const float* in,
                                     float* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);

// Double-precision counterpart to :cpp:func:`permute_copy_f32`.
//
// Parameters
// ----------
// in : const double*
//     Source buffer laid out densely (C-order) according to ``in_shape``.
// out : double*
//     Pre-allocated destination buffer of ``numel(in_shape) * sizeof(double)``
//     bytes.
// in_shape : const std::vector<int64_t>&
//     Shape of the source tensor.
// perm : const std::vector<int>&
//     Axis permutation; ``perm[d]`` names the source axis whose extent becomes
//     output axis ``d``.
LUCID_INTERNAL void permute_copy_f64(const double* in,
                                     double* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);

// Int32 counterpart to :cpp:func:`permute_copy_f32`.
//
// Used for argmax / index tensors that need to be re-laid out as part of a
// larger op (e.g. transposed indexing).
//
// Parameters
// ----------
// in : const int32_t*
//     Source buffer laid out densely (C-order) according to ``in_shape``.
// out : int32_t*
//     Pre-allocated destination buffer.
// in_shape : const std::vector<int64_t>&
//     Shape of the source tensor.
// perm : const std::vector<int>&
//     Axis permutation; see :cpp:func:`permute_copy_f32`.
LUCID_INTERNAL void permute_copy_i32(const std::int32_t* in,
                                     std::int32_t* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);

// Two-byte counterpart to :cpp:func:`permute_copy_f32`, shared by ``int16``
// and ``float16`` — permutation moves bytes, so the width is what matters.
LUCID_INTERNAL void permute_copy_i16(const std::int16_t* in,
                                     std::int16_t* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);

// One-byte counterpart, shared by ``int8`` and ``bool``.
LUCID_INTERNAL void permute_copy_i8(const std::int8_t* in,
                                    std::int8_t* out,
                                    const std::vector<std::int64_t>& in_shape,
                                    const std::vector<int>& perm);

// Int64 counterpart to :cpp:func:`permute_copy_f32`.
//
// Parameters
// ----------
// in : const int64_t*
//     Source buffer laid out densely (C-order) according to ``in_shape``.
// out : int64_t*
//     Pre-allocated destination buffer.
// in_shape : const std::vector<int64_t>&
//     Shape of the source tensor.
// perm : const std::vector<int>&
//     Axis permutation; see :cpp:func:`permute_copy_f32`.
LUCID_INTERNAL void permute_copy_i64(const std::int64_t* in,
                                     std::int64_t* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);

}  // namespace lucid::backend::cpu
