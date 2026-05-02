#pragma once

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
    return storage_gpu(s);
}
inline const GpuStorage& gpu_get(const Storage& s) {
    return storage_gpu(s);
}

inline ::mlx::core::array mlx_scalar(double x, Dtype dt) {
    return gpu::mlx_scalar(x, gpu::to_mlx_dtype(dt));
}

inline void gpu_replace(GpuStorage& dst, ::mlx::core::array&& arr, Dtype dt) {
    dst.arr = gpu::wrap_mlx_array(std::move(arr), dt).arr;
}

}  // namespace lucid::optim_detail
