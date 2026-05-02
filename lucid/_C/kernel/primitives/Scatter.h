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

inline void zero_cpu_storage(CpuStorage& s) {
    if (s.ptr && s.nbytes)
        std::memset(s.ptr.get(), 0, s.nbytes);
}

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
