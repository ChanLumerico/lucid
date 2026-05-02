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

inline int make_nearest_scale_index(int out_idx, int in_dim, int out_dim) {
    const int idx = static_cast<int>(std::floor(static_cast<double>(out_idx) * in_dim / out_dim));
    return std::clamp(idx, 0, in_dim - 1);
}

}  // namespace primitives
}  // namespace kernel
}  // namespace lucid
