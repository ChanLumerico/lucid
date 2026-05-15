// lucid/_C/kernel/primitives/Gather.h
//
// CPU gather and nearest-neighbor index helpers used by the Embedding
// forward pass and spatial upsampling (nearest-neighbor interpolation).
// These are intentionally low-level, typed functions rather than
// Storage-level wrappers so that higher-level ops can compose them with
// custom index precomputation without double allocations.

#pragma once

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

// Read n_out elements from src_data at the positions given by flat_indices,
// writing results sequentially into dst_data. flat_indices[i] is the
// linearised (row-major) offset into src_data for output position i.
// The caller is responsible for ensuring all indices are in-bounds.
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

// Map an output spatial index to the corresponding nearest input index
// under the nearest-neighbor scaling convention. The formula is
// floor(out_idx * in_dim / out_dim), clamped to [0, in_dim - 1] to
// handle rounding at the boundary.
inline int make_nearest_scale_index(int out_idx, int in_dim, int out_dim) {
    const int idx = static_cast<int>(std::floor(static_cast<double>(out_idx) * in_dim / out_dim));
    return std::clamp(idx, 0, in_dim - 1);
}

}  // namespace primitives
}  // namespace kernel
}  // namespace lucid
