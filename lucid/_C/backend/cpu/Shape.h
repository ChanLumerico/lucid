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

// Copies a tensor of ``elem``-byte elements into a permuted, densely-packed
// output buffer using the supplied axis permutation.
//
// The output is the input viewed through its C-order strides reordered by
// ``perm``, packed by :func:`strided::pack` — axes merged, contiguous runs
// moved by memcpy, a merged 2-D transpose copied in tiles.  This is an
// out-of-place layout transform — there is no in-place fast path.
//
// Permutation moves bytes and never reads a value, so one entry point keyed
// by element width serves every dtype.  It used to be one function per
// element type, and the backend's switch over them named float16 but not
// bfloat16 and had no 8- or 16-byte case for complex, so the backward of a
// transpose refused those dtypes on the CPU while Metal ran them.
//
// Parameters
// ----------
// in : const std::byte*
//     Source buffer laid out densely (C-order) according to ``in_shape``.
// out : std::byte*
//     Destination buffer.  Caller must pre-allocate
//     ``numel(in_shape) * elem`` bytes.  May not alias ``in``.
// in_shape : const std::vector<int64_t>&
//     Shape of the source tensor.  An empty vector is treated as a 0-D scalar
//     and produces a single-element copy.
// perm : const std::vector<int>&
//     Axis permutation of length ``in_shape.size()``.  ``perm[d]`` names the
//     source axis whose extent becomes output axis ``d``.  Must be a valid
//     permutation of ``[0, ndim)``.
// elem : std::size_t
//     Element width in bytes (``dtype_size``).
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
LUCID_INTERNAL void permute_copy(const std::byte* in,
                                 std::byte* out,
                                 const std::vector<std::int64_t>& in_shape,
                                 const std::vector<int>& perm,
                                 std::size_t elem);

}  // namespace lucid::backend::cpu
