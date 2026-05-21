// lucid/_C/backend/cpu/Shape.h
//
// CPU shape-transformation helper: permute_copy performs an N-D transpose by
// copying elements in the permuted order into a fresh densely-packed buffer.
// This is used by CpuBackend::permute_cpu() and by the GPU backend's tensordot
// data-layout preparation path.
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
// The implementation iterates the output in flat row-major order and, for
// each output flat index, back-computes its N-D coordinate and reads from
// the matching input position via the inverse permutation expressed through
// the input's C-order strides.  No intermediate allocation beyond two stride
// vectors of size ``ndim`` is performed.  This is an out-of-place layout
// transform — there is no in-place fast path.
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
// Cost is ``O(numel * ndim)`` due to per-element coordinate back-projection.
// For frequently used permutations (e.g. matrix transpose) consider a
// dimension-specialised kernel; this routine is the generic fallback used by
// :cpp:class:`CpuBackend` and the GPU tensordot pre-layout path.
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
