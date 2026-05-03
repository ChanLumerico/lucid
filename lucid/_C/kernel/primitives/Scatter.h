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

// Accumulate n_contrib weighted source values into dst at scattered
// positions. For each i: dst[flat_dst_indices[i]] += src[i] * weights[i].
// Multiple contributions to the same dst index are summed, which is the
// correct adjoint for a forward gather.
// dst must be pre-zeroed by the caller when building a fresh gradient.
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

// Zero all bytes of the given CpuStorage in-place.
inline void zero_cpu_storage(CpuStorage& s) {
    if (s.ptr && s.nbytes)
        std::memset(s.ptr.get(), 0, s.nbytes);
}

// Allocate a new zero-filled CpuStorage of numel elements with dtype dt.
// Convenience wrapper used when creating gradient accumulation buffers.
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
