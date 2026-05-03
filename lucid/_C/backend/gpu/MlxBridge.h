// lucid/_C/backend/gpu/MlxBridge.h
//
// Conversion helpers between Lucid's Storage types and mlx::core::array.
//
// Critical constraint: mlx::core::allocator::malloc() allocates GPU-private
// pages on Apple Silicon Metal.  The raw data() pointer of an unevaluated
// or GPU-allocated mlx array must NEVER be accessed from the CPU — doing so
// causes SIGBUS on M-series hardware.  The upload_cpu_to_gpu function uses
// mlx::core::copy() (a lazy DRAM copy at ~100 GB/s) to ensure the resulting
// array owns GPU-accessible memory without aliasing the CPU buffer.
//
// GpuStorage holds a shared_ptr<mlx::core::array> so that multiple Storage
// objects can share the same underlying MLX array without copying.  The
// make_tracked helper in the .cpp registers each allocation with MemoryTracker
// so Python-side memory reporting remains accurate.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>

#include "../../api.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

// Converts a Lucid Dtype to the corresponding mlx::core::Dtype.
// Throws not_implemented for Dtype::F64 because Metal/MLX does not support
// 64-bit floating-point on the GPU.
::mlx::core::Dtype to_mlx_dtype(Dtype dt);

// Converts an mlx::core::Dtype back to the Lucid Dtype.
// Throws not_implemented for uint* and bfloat16 which have no Lucid equivalent.
Dtype from_mlx_dtype(::mlx::core::Dtype dt);

// Uploads a CPU buffer to GPU memory via mlx::core::copy().  The copy is
// necessary because mlx arrays own GPU-private memory that cannot alias the
// CPU pointer.  The CpuStorage shared_ptr is kept alive as a custom deleter
// on the external MLX array until after the copy completes.
LUCID_API GpuStorage upload_cpu_to_gpu(const CpuStorage& cpu, const Shape& shape);

// Wraps a SharedStorage (MTLResourceStorageModeShared) buffer as a GpuStorage
// without copying.  The resulting array is backed by the shared Metal buffer
// and can be read/written by both CPU and GPU.
LUCID_API GpuStorage shared_storage_to_gpu(const SharedStorage& sh, const Shape& shape);

// Downloads a GPU array to CPU by calling arr.eval() (forces materialisation)
// and then memcpy-ing the data pointer to a newly allocated CPU buffer.
LUCID_API CpuStorage download_gpu_to_cpu(const GpuStorage& gpu, const Shape& shape);

// Wraps an already-owned mlx::core::array (moved in) into a GpuStorage that
// participates in MemoryTracker accounting.
GpuStorage wrap_mlx_array(::mlx::core::array&& arr, Dtype dtype);

// Converts a Lucid Shape (std::vector<int64_t>) to mlx::core::Shape
// (std::vector<int32_t>).  Throws if any dimension exceeds INT32_MAX.
::mlx::core::Shape to_mlx_shape(const Shape& shape);

// Converts an mlx::core::Shape to a Lucid Shape.
inline Shape mlx_shape_to_lucid(const ::mlx::core::Shape& shape) {
    Shape out;
    out.reserve(shape.size());
    for (auto dim : shape)
        out.push_back(static_cast<std::int64_t>(dim));
    return out;
}

// Creates a scalar mlx array of dtype dt with value v.  Used as a broadcast
// operand in GpuBackend arithmetic helpers.
inline ::mlx::core::array mlx_scalar(double v, ::mlx::core::Dtype dt) {
    return ::mlx::core::astype(::mlx::core::array(static_cast<float>(v)), dt);
}

}  // namespace lucid::gpu
