// lucid/_C/kernel/primitives/Scatter.h
//
// CPU scatter-accumulate primitives used in backward passes for ops that
// perform index-based reads (Embedding, gather, upsampling). When the
// forward pass reads from scattered positions, the backward pass must
// accumulate gradients into those same positions via scatter-add.
// Also provides utility functions for zeroing and allocating zero-filled
// CpuStorage objects, which are needed when constructing gradient buffers
// that accumulate into initially-zero memory.

#pragma once

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

// Weighted scatter-add into a destination buffer at scattered positions.
//
// Adjoint of :func:`gather_nd_cpu`: every value of ``src`` is multiplied
// by the corresponding ``weights`` entry and added into ``dst`` at the
// flat position ``flat_dst_indices[i]``.  Duplicate indices accumulate
// by summation — the mathematically correct gradient transform whenever
// the forward gather read the same source position more than once
// (e.g., shared embedding rows, overlapping upsample windows).
//
// Math
// ----
// For each contribution index $i$:
// $$
//   \text{dst}[\,\text{flat\_dst\_indices}[i]\,]
//   \mathrel{+}= \text{src}[i] \cdot \text{weights}[i]
// $$
// Multiple $i$ mapping to the same destination index sum together.
//
// Parameters
// ----------
// dst : T*
//     Destination buffer.  Must be pre-zeroed by the caller when used
//     to build a fresh gradient — this routine only accumulates.
// src : const T*
//     Source values to scatter; one entry per contribution.
// flat_dst_indices : const std::vector<std::size_t>&
//     Row-major flat offsets into ``dst``.  Size must be at least
//     ``n_contrib``.
// weights : const std::vector<T>&
//     Per-contribution scalar weight (e.g., 1.0 for plain scatter, or
//     interpolation coefficients for bilinear backwards).
// n_contrib : std::size_t
//     Number of (src, index, weight) triplets to apply.
//
// Notes
// -----
// CPU-only routine — single-threaded, no atomics.  The GPU equivalent
// uses MLX's native scatter primitive which performs atomic
// accumulation across threads.  ``dst`` ownership remains with the
// caller; this function does not allocate.
//
// See Also
// --------
// gather_nd_cpu : The forward gather whose adjoint this implements.
// make_zero_cpu_storage : Convenience allocator for a pre-zeroed
//     gradient buffer.
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

// Zero all bytes of a CpuStorage in-place.
//
// Convenience wrapper around ``std::memset`` that handles the
// edge cases of a null pointer or zero-sized buffer.  Used to reset
// gradient accumulation buffers between mini-batches or before a
// scatter-add pass.
//
// Parameters
// ----------
// s : CpuStorage&
//     Storage to zero.  No-op if ``s.ptr`` is null or ``s.nbytes == 0``.
//
// Notes
// -----
// Operates on the raw byte buffer regardless of dtype; safe for every
// numeric :data:`Dtype` since IEEE-754 zero and integer zero both have
// all-zero byte representations.
inline void zero_cpu_storage(CpuStorage& s) {
    if (s.ptr && s.nbytes)
        std::memset(s.ptr.get(), 0, s.nbytes);
}

// Allocate a new zero-filled CpuStorage.
//
// Convenience constructor used when building gradient accumulation
// buffers: allocates an aligned byte buffer sized for ``numel`` elements
// of dtype ``dt`` and clears it to zero so that subsequent scatter-add
// calls accumulate against a clean slate.
//
// Parameters
// ----------
// numel : std::size_t
//     Number of elements (not bytes) to allocate.
// dt : Dtype
//     Element dtype; determines the per-element byte size via
//     :func:`dtype_size`.
//
// Returns
// -------
// CpuStorage
//     Newly allocated, aligned, fully zero-filled storage with
//     ``nbytes = numel * dtype_size(dt)``.
//
// See Also
// --------
// zero_cpu_storage : In-place re-zero of an existing storage.
// scatter_add_cpu  : Primary consumer of the zero-initialised buffer.
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
