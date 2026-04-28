#pragma once

// =====================================================================
// Lucid C++ engine — internal optimizer helpers.
// =====================================================================
//
// Shared inline helpers used by every optimizer .cpp:
//   - typed CPU pointer extraction
//   - GPU storage variant access
//   - MLX scalar construction in the right dtype
//   - in-place GpuStorage replacement (functional MLX semantics)
//
// Intentionally header-internal (not exposed by Optimizer.h) — only the
// optimizer .cpp files include it. Filename prefix `_` signals that this
// is an internal-only header.

#include <cstddef>
#include <utility>
#include <variant>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace lucid::optim_detail {

template <typename T>
inline T* cpu_ptr(Storage& s) {
    return reinterpret_cast<T*>(std::get<CpuStorage>(s).ptr.get());
}

template <typename T>
inline const T* cpu_cptr(const Storage& s) {
    return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
}

inline std::size_t cpu_numel(const TensorImpl& t) {
    return t.numel();
}

inline GpuStorage& gpu_get(Storage& s) {
    return std::get<GpuStorage>(s);
}
inline const GpuStorage& gpu_get(const Storage& s) {
    return std::get<GpuStorage>(s);
}

inline ::mlx::core::array mlx_scalar(double x, Dtype dt) {
    return ::mlx::core::array(x, gpu::to_mlx_dtype(dt));
}

inline void gpu_replace(GpuStorage& dst, ::mlx::core::array&& arr, Dtype dt) {
    dst.arr = gpu::wrap_mlx_array(std::move(arr), dt).arr;
}

}  // namespace lucid::optim_detail
