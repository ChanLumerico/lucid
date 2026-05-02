#pragma once

// =====================================================================
// Lucid C++ engine — MLX bridge (Phase 3.7).
// =====================================================================
//
// Centralizes:
//   - Lucid Dtype  ↔  mlx::core::Dtype  conversion
//   - CpuStorage   →  mlx::core::array  upload  (host-to-device copy)
//   - mlx::core::array → CpuStorage     download (device-to-host eval+copy)
//   - allocation accounting against MemoryTracker(GPU)
//
// Anything that needs to move data between Lucid's CPU storage and an
// MLX array goes through this header. Op-level files (Add.cpp etc.)
// include `<mlx/ops.h>` directly for compute, but never reach into
// allocator details.
//
// Layer: backend/gpu/. Pulls in mlx headers; only included from .cpp
// files inside the engine library, never from public headers.

#include <cstddef>
#include <memory>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>  // for ::mlx::core::astype used by mlx_scalar()

#include "../../api.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

// Lucid Dtype → MLX Dtype.
::mlx::core::Dtype to_mlx_dtype(Dtype dt);

// MLX Dtype → Lucid Dtype. Throws on unsupported values.
Dtype from_mlx_dtype(::mlx::core::Dtype dt);

// Build an mlx::core::array from a Lucid CpuStorage. The MLX array is
// constructed via `mlx::core::array(data_ptr, shape, dtype, deleter)` —
// MLX may attempt to use the buffer without a copy, so we keep the
// CpuStorage shared_ptr alive via a deleter closure. Shape is copied.
/// Upload cpu to gpu.
/// If cpu was allocated through MetalAllocator::allocate_shared, MLX wraps the
/// existing Metal buffer without a copy (Phase 9.3 zero-copy path).
LUCID_API GpuStorage upload_cpu_to_gpu(const CpuStorage& cpu, const Shape& shape);

/// Phase 9.3: promote a SharedStorage directly to GpuStorage with zero copy.
/// The returned GpuStorage shares the same physical memory as the input.
LUCID_API GpuStorage shared_storage_to_gpu(const SharedStorage& sh, const Shape& shape);

// Download a GPU array back to a freshly-allocated CpuStorage. Calls
// `arr.eval()` first so subsequent reads are safe. Result is C-contiguous.
/// Download gpu to cpu.
LUCID_API CpuStorage download_gpu_to_cpu(const GpuStorage& gpu, const Shape& shape);

// Wrap an existing mlx::core::array as a GpuStorage. Used by GPU op
// kernels after compute. `nbytes` is recorded for MemoryTracker.
GpuStorage wrap_mlx_array(::mlx::core::array&& arr, Dtype dtype);

// Convert a Lucid Shape (int64) to an MLX shape (int32). Throws if any
// dim exceeds INT32_MAX.
::mlx::core::Shape to_mlx_shape(const Shape& shape);

// Convert an MLX shape back to Lucid's int64 shape representation.
inline Shape mlx_shape_to_lucid(const ::mlx::core::Shape& shape) {
    Shape out;
    out.reserve(shape.size());
    for (auto dim : shape)
        out.push_back(static_cast<std::int64_t>(dim));
    return out;
}

// Build a 0-dim `mlx::array` holding `v` cast to `dt`. Replaces ad-hoc
// `mlx_scalar`/`mlx_scalar_dt` copies that used to live in nn / optim ops.
inline ::mlx::core::array mlx_scalar(double v, ::mlx::core::Dtype dt) {
    return ::mlx::core::astype(::mlx::core::array(static_cast<float>(v)), dt);
}

}  // namespace lucid::gpu
