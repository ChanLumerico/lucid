#pragma once

// =====================================================================
// Lucid C++ engine — kernel/primitives/Gather.h
// =====================================================================
//
// Primitive: N-D coordinate gather (nearest-neighbor flat-index lookup).
//
// Conceptually:
//   Given a source tensor of shape [..., D0, D1, ..., Dk] and a set of
//   integer index arrays (one per spatial dimension), gather values from
//   the source at those coordinates and produce an output of shape
//   [..., out_D0, out_D1, ..., out_Dk].
//
// Two flavors are present in the codebase:
//
// CPU path (hand-rolled loops):
//   Each output coordinate is computed via a scale formula
//       src_idx[i] = floor(out_idx[i] * in_dim[i] / out_dim[i])
//   then clamped to [0, in_dim-1], and used to look up the flat source
//   offset. Found in:
//     - nn/Interpolate.cpp: nearest2d_cpu / nearest3d_cpu
//     - nn/Vision.cpp: rotate / nearest_2d / nearest_3d
//
// GPU path (MLX flat-index take):
//   Per-axis integer index arrays are built with:
//       idx = floor(arange(out_dim) * (in_dim / out_dim))
//       idx = clip(astype(idx, int64), 0, in_dim - 1)
//   Then combined into a flat offset via stride arithmetic and passed to
//   mlx::core::take on the flattened input buffer. Found in:
//     - nn/Interpolate.cpp: interpolate_nearest_{2d,3d}_op (GPU branch)
//     - nn/Vision.cpp: rotate_op / nearest_2d_op / nearest_3d_op (GPU)
//     - nn/Interpolate.cpp: gather_2d_corner (bilinear corners on GPU)
//
// Design note:
//   The CPU implementations are tightly coupled to their specific ops
//   (nearest-2D, nearest-3D, bilinear-corner) and do NOT go through a
//   shared runtime helper — they keep their local templated loops for
//   maximum cache efficiency. This header declares the canonical
//   primitive contract and provides a thin gather_nd_cpu helper for the
//   simplest case (single nearest-neighbor lookup per output element).
//
// Layer: kernel/primitives/. No extra includes beyond core/; callers own
//        their source buffers.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Storage.h"

namespace lucid {
namespace kernel {
namespace primitives {

// ---------------------------------------------------------------------
// gather_nd_cpu
// ---------------------------------------------------------------------
//
// Flat-index gather from a contiguous source tensor given a precomputed
// vector of flat destination indices.
//
//   src_data   — pointer to the source buffer (row-major, contiguous)
//   dst_data   — pointer to the output buffer (row-major, contiguous)
//   flat_indices — flat linear index into src for each output element
//   n_out      — total number of output elements
//
// Template parameter T must be a scalar (float or double).
//
// The caller is responsible for:
//   - pre-computing flat_indices (e.g. from per-axis coords and strides)
//   - allocating dst_data (n_out * sizeof(T) bytes)
//   - ensuring all flat_indices are in [0, src_numel).
//
template <typename T>
inline void gather_nd_cpu(const T* src_data,
                          T* dst_data,
                          const std::vector<std::size_t>& flat_indices,
                          std::size_t n_out) {
    assert(flat_indices.size() == n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
        dst_data[i] = src_data[flat_indices[i]];
    }
}

// ---------------------------------------------------------------------
// make_nearest_scale_index
// ---------------------------------------------------------------------
//
// Utility: compute the nearest-neighbor source index for a single
// output coordinate, mirroring the formula used in nearest2d_cpu /
// nearest3d_cpu.
//
//   out_idx   — output coordinate in [0, out_dim)
//   in_dim    — source spatial dimension
//   out_dim   — output spatial dimension
//
// Returns: clamped integer source coordinate in [0, in_dim - 1].
//
inline int make_nearest_scale_index(int out_idx, int in_dim, int out_dim) {
    const int idx = static_cast<int>(std::floor(static_cast<double>(out_idx) * in_dim / out_dim));
    return std::clamp(idx, 0, in_dim - 1);
}

}  // namespace primitives
}  // namespace kernel
}  // namespace lucid

// -----------------------------------------------------------------------
// GPU gather pattern (documentation only — not compiled here)
// -----------------------------------------------------------------------
//
// The canonical MLX flat-index gather pattern used by nearest-2D/3D
// interpolation and the bilinear corner gather is:
//
//   // 1. Build per-axis int64 indices (one arange per spatial dim):
//   auto h_idx = astype(floor(multiply(arange(H_out), scale_H)), int64);
//   h_idx = clip(h_idx, optional(zero_i), optional(Hm1_i));
//
//   // 2. Compute flat index via stride arithmetic (broadcast-friendly):
//   //    flat = n * (C*H*W) + c * (H*W) + h * W + w
//   auto flat = add(add(multiply(n_idx, sN), multiply(c_idx, sC)),
//                   add(multiply(h_b, sH), w_b));
//   auto flat_b = broadcast_to(flat, {N, C, H_out, W_out});
//
//   // 3. Gather via mlx::core::take on a flattened input view:
//   auto in_flat = reshape(input, {N * C * H_in * W_in});
//   auto result  = take(in_flat, flat_b);
//
// For multi-corner gathers (bilinear/trilinear), this pattern is applied
// once per corner (4 corners for 2D, 8 corners for 3D) and the results
// are combined with interpolation weights.
//
// See: nn/Interpolate.cpp gather_2d_corner, interpolate_nearest_{2d,3d}_op
//      nn/Vision.cpp (GPU branch)
