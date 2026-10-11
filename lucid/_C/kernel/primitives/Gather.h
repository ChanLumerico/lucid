// lucid/_C/kernel/primitives/Gather.h
//
// CPU gather helpers used by the Embedding forward pass.  These are
// intentionally low-level, typed functions rather than Storage-level
// wrappers so that higher-level ops can compose them with custom index
// precomputation without double allocations.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Storage.h"

namespace lucid {
namespace kernel {
namespace primitives {

// Gather elements from a source buffer at precomputed flat positions.
//
// Reads ``n_out`` values from ``src_data`` at the row-major offsets in
// ``flat_indices`` and writes them sequentially into ``dst_data``.  This
// is the inner loop of every index-select / embedding-lookup operation:
// callers pre-compute the flat index map once and reuse it across the
// gather plus the corresponding scatter in the backward pass.
//
// Math
// ----
// For every output position $i$:
// $$
//   \text{dst}[i] = \text{src}[\text{flat\_indices}[i]]
// $$
//
// Parameters
// ----------
// src_data : const T*
//     Pointer to the contiguous source buffer.
// dst_data : T*
//     Pointer to the contiguous destination buffer with at least
//     ``n_out`` writable elements.
// flat_indices : const std::vector<std::size_t>&
//     Row-major flat offsets into ``src_data``; one entry per output
//     element.  Size must equal ``n_out``.
// n_out : std::size_t
//     Number of elements to gather.
//
// Notes
// -----
// The caller is responsible for ensuring every index is in-bounds.  The
// function performs no bounds checking on release builds — only an
// assertion that the index vector length matches ``n_out``.
//
// See Also
// --------
// scatter_add_cpu : The adjoint operation used in the backward pass.
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

}  // namespace primitives
}  // namespace kernel
}  // namespace lucid
