#pragma once

#include <cstddef>
#include <memory>
#include <variant>

#include "Dtype.h"

// Forward-declare mlx::core::array so this header doesn't pull in MLX's
// transitive include set on every translation unit. Files that actually
// touch the MLX array (only `backend/gpu/MlxBridge.cpp` and a handful of
// dispatch sites) include `<mlx/array.h>` themselves.
namespace mlx::core {
class array;
}

namespace lucid {

struct CpuStorage {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
};

// Phase 3.7: GPU storage holds a shared_ptr to an mlx::core::array. MLX
// arrays are inherently shareable (CoW desc internally), so wrapping in
// shared_ptr keeps copy/move of Storage cheap and lets multiple TensorImpls
// observe the same MLX array without redundant allocations. The `dtype`
// and `nbytes` fields mirror CpuStorage so dispatch sites that only need
// metadata don't have to peek into the MLX array.
struct GpuStorage {
    std::shared_ptr<mlx::core::array> arr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
};

using Storage = std::variant<CpuStorage, GpuStorage>;

// --------------------------------------------------------------------------- //
// Storage free-function helpers — avoid repeated std::visit boilerplate.
// --------------------------------------------------------------------------- //

inline bool storage_is_cpu(const Storage& s) noexcept {
    return s.index() == 0;
}
inline bool storage_is_gpu(const Storage& s) noexcept {
    return s.index() == 1;
}
inline std::size_t storage_nbytes(const Storage& s) noexcept {
    return s.index() == 0 ? std::get<0>(s).nbytes : std::get<1>(s).nbytes;
}
inline Dtype storage_dtype(const Storage& s) noexcept {
    return s.index() == 0 ? std::get<0>(s).dtype : std::get<1>(s).dtype;
}

}  // namespace lucid
