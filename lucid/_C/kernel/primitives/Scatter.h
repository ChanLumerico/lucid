#pragma once

// =====================================================================
// Lucid C++ engine — kernel/primitives/Scatter.h
// =====================================================================
//
// Primitive: multi-corner weighted scatter-add (accumulate into output).
//
// Conceptually:
//   For each output position (h, w, ...) the primitive distributes a
//   gradient value g to K "corner" positions in the input/source tensor,
//   each weighted by a bilinear/trilinear interpolation coefficient:
//
//       dst[corner_coord_k] += g * weight_k   for k in 0..K-1
//
//   This is the adjoint of a weighted multi-corner gather (forward
//   interpolation). The output buffer is first zeroed, then each output
//   element accumulates its contribution atomically into the input-shape
//   gradient buffer.
//
// Where the pattern appears:
//
//   CPU path (hand-rolled atomic accumulate):
//     - nn/Interpolate.cpp: bilinear_backward_cpu  (4 corners, 2-D)
//     - nn/Interpolate.cpp: trilinear_backward_cpu (8 corners, 3-D)
//     - nn/Spatial.cpp:     GridSampleBackward::apply (4 corners, bilinear)
//   The CPU implementations use a direct
//       base[y * W + x] += g * weight;
//   accumulation loop (safe since CPU is single-threaded per slice).
//
//   GPU path (MLX scatter_add):
//     - nn/Interpolate.cpp: InterpolateBilinearBackward::apply  (GPU)
//     - nn/Interpolate.cpp: InterpolateTrilinearBackward::apply (GPU)
//     - nn/Spatial.cpp:     GridSampleBackward::apply           (GPU)
//   The GPU uses mlx::core::scatter_add with index arrays computed via
//   the same stride arithmetic as the forward gather. Each corner
//   becomes one scatter_add call that accumulates into a zero-filled
//   output buffer.
//
// Design note:
//   The full backward implementations are complex (they differ in
//   dimensionality, padding mode, alignment, and whether dgrid is also
//   needed) and remain in their respective .cpp files. This header
//   defines the canonical primitive interface: scatter_add_cpu, and
//   documents the GPU contract so new ops can follow the same pattern.
//
// Layer: kernel/primitives/. No extra includes beyond core/; callers own
//        their buffers.

#include <cstddef>
#include <cstring>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Storage.h"

namespace lucid {
namespace kernel {
namespace primitives {

// ---------------------------------------------------------------------
// scatter_add_cpu
// ---------------------------------------------------------------------
//
// Accumulate weighted contributions from one "corner" into a flat
// destination buffer. This is the single-corner building block; call
// once per interpolation corner (4× for bilinear, 8× for trilinear).
//
// Parameters:
//   dst        — destination (input-gradient) buffer, already zeroed.
//                Shape: [dst_numel] elements of type T.
//   src        — gradient-output (upstream grad) buffer, flat.
//                Shape: [n_contrib] elements.
//   flat_dst_indices — flat destination index for each contribution.
//                      Length: n_contrib. All values in [0, dst_numel).
//   weights    — per-element interpolation weight. Length: n_contrib.
//   n_contrib  — number of (src, dst, weight) triples to process.
//
// Effect:
//   dst[flat_dst_indices[i]] += src[i] * weights[i]  for i in [0, n_contrib)
//
// Thread safety: single-threaded (no atomic ops). Use per-batch-slice
// parallelism at a higher level if needed.
//
template <typename T>
inline void scatter_add_cpu(T* dst,
                            const T* src,
                            const std::vector<std::size_t>& flat_dst_indices,
                            const std::vector<T>& weights,
                            std::size_t n_contrib) {
    for (std::size_t i = 0; i < n_contrib; ++i) {
        dst[flat_dst_indices[i]] += src[i] * weights[i];
    }
}

// ---------------------------------------------------------------------
// zero_cpu_storage
// ---------------------------------------------------------------------
//
// Zero-fill a CpuStorage in preparation for scatter-add accumulation.
// Callers should invoke this before the first scatter_add_cpu call.
//
inline void zero_cpu_storage(CpuStorage& s) {
    if (s.ptr && s.nbytes)
        std::memset(s.ptr.get(), 0, s.nbytes);
}

// ---------------------------------------------------------------------
// make_zero_cpu_storage
// ---------------------------------------------------------------------
//
// Allocate and zero a CpuStorage of numel * sizeof(dtype) bytes.
// Convenience wrapper for the common "allocate gradient buffer" pattern
// in backward passes.
//
inline CpuStorage make_zero_cpu_storage(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    std::memset(s.ptr.get(), 0, s.nbytes);
    return s;
}

}  // namespace primitives
}  // namespace kernel
}  // namespace lucid

// -----------------------------------------------------------------------
// GPU scatter-add pattern (documentation only — not compiled here)
// -----------------------------------------------------------------------
//
// The canonical MLX scatter_add pattern used by bilinear/trilinear/grid-
// sample backward passes (one call per interpolation corner):
//
//   // Bilinear 2-D: 4 corners on base shape [N, C, H_in, W_in]
//   ::mlx::core::Shape base_shape{N, C, H_in, W_in};
//   auto base = ::mlx::core::zeros(base_shape, mlx_dt);
//   std::vector<int> axes_v{0, 1, 2, 3};
//
//   auto scatter_one = [&](const array& y_idx, const array& x_idx,
//                          const array& weight) {
//       // index arrays must be broadcast-compatible with base_shape
//       std::vector<array> idxs{n_idx, c_idx, y_idx, x_idx};
//       // updates shape = base_shape + trailing 1 per scatter axis
//       auto upd = reshape(multiply(grad_out, weight),
//                          {N, C, H_out, W_out, 1, 1, 1, 1});
//       base = scatter_add(base, idxs, upd, axes_v);
//   };
//   scatter_one(y0_b, x0_b, w00);  // top-left corner
//   scatter_one(y0_b, x1_b, w01);  // top-right
//   scatter_one(y1_b, x0_b, w10);  // bottom-left
//   scatter_one(y1_b, x1_b, w11);  // bottom-right
//
// For grid_sample backward (flat-buffer variant):
//   // Accumulate into a 1-D dinput_flat = zeros({N*C*H*W}).
//   auto flat_idx_1d = reshape(flat_idx, {N*C*H_out*W_out});
//   auto contrib_2d  = reshape(multiply(go, w), {N*C*H_out*W_out, 1});
//   dinput_flat = scatter_add(dinput_flat, flat_idx_1d, contrib_2d, 0);
//
// See: nn/Interpolate.cpp InterpolateBilinearBackward::apply (GPU)
//      nn/Interpolate.cpp InterpolateTrilinearBackward::apply (GPU)
//      nn/Spatial.cpp     GridSampleBackward::apply (GPU)
