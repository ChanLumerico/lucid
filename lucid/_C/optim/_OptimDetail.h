// lucid/_C/optim/_OptimDetail.h
//
// Internal helpers shared by all optimizer implementations. Provides
// thin, inline wrappers around Storage accessors and MLX scalar
// construction so that the optimizer .cpp files stay concise and
// avoid repeating the same casts and variant extractions. Nothing in
// this header is part of the public API.

#pragma once

#include <cstddef>
#include <utility>
#include <variant>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace lucid::optim_detail {

// Return a typed mutable pointer into the raw bytes of a CpuStorage.
// The caller must ensure T matches the storage's Dtype.
template <typename T>
inline T* cpu_ptr(Storage& s) {
    return reinterpret_cast<T*>(std::get<CpuStorage>(s).ptr.get());
}

// Return a typed const pointer into the raw bytes of a CpuStorage.
template <typename T>
inline const T* cpu_cptr(const Storage& s) {
    return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
}

// Return the total element count of a tensor's CPU storage.
inline std::size_t cpu_numel(const TensorImpl& t) {
    return t.numel();
}

// Extract the GpuStorage variant from a mutable Storage.
inline GpuStorage& gpu_get(Storage& s) {
    return storage_gpu(s);
}

// Extract the GpuStorage variant from a const Storage.
inline const GpuStorage& gpu_get(const Storage& s) {
    return storage_gpu(s);
}

// Construct a zero-dimensional MLX scalar array with the given dtype.
// Used to form broadcast-compatible scalar operands in GPU update rules.
inline ::mlx::core::array mlx_scalar(double x, Dtype dt) {
    return gpu::mlx_scalar(x, gpu::to_mlx_dtype(dt));
}

// Replace the MLX array inside dst with the result of a GPU computation.
// Ownership of arr is transferred; the Lucid dtype tag is preserved.
inline void gpu_replace(GpuStorage& dst, ::mlx::core::array&& arr, Dtype dt) {
    dst.arr = gpu::wrap_mlx_array(std::move(arr), dt).arr;
}

}  // namespace lucid::optim_detail
