#include "MlxBridge.h"

#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include <mlx/ops.h>

#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/MemoryStats.h"

namespace lucid::gpu {

::mlx::core::Dtype to_mlx_dtype(Dtype dt) {
    switch (dt) {
        case Dtype::Bool: return ::mlx::core::bool_;
        case Dtype::I8:   return ::mlx::core::int8;
        case Dtype::I16:  return ::mlx::core::int16;
        case Dtype::I32:  return ::mlx::core::int32;
        case Dtype::I64:  return ::mlx::core::int64;
        case Dtype::F16:  return ::mlx::core::float16;
        case Dtype::F32:  return ::mlx::core::float32;
        case Dtype::F64:
            // MLX-Metal does not support float64 on the GPU. Reject early
            // with a clear, actionable message rather than letting MLX raise
            // a generic ValueError deep inside its eval() pipeline.
            throw NotImplementedError(
                "float64 is not supported on GPU (MLX-Metal limitation). "
                "Cast to float32 first, or keep the tensor on CPU.");
        case Dtype::C64:  return ::mlx::core::complex64;
    }
    throw LucidError("to_mlx_dtype: unknown Dtype");
}

Dtype from_mlx_dtype(::mlx::core::Dtype dt) {
    using V = ::mlx::core::Dtype::Val;
    switch (dt.val()) {
        case V::bool_:    return Dtype::Bool;
        case V::int8:     return Dtype::I8;
        case V::int16:    return Dtype::I16;
        case V::int32:    return Dtype::I32;
        case V::int64:    return Dtype::I64;
        case V::float16:  return Dtype::F16;
        case V::float32:  return Dtype::F32;
        case V::float64:  return Dtype::F64;
        case V::complex64: return Dtype::C64;
        default:
            throw NotImplementedError(
                "from_mlx_dtype: unsupported MLX dtype "
                "(uint*/bfloat16 not yet wired into Lucid)");
    }
}

::mlx::core::Shape to_mlx_shape(const Shape& shape) {
    ::mlx::core::Shape out;
    out.reserve(shape.size());
    for (auto d : shape) {
        if (d > std::numeric_limits<::mlx::core::ShapeElem>::max()) {
            throw LucidError(
                "to_mlx_shape: dimension exceeds int32 range (MLX requires "
                "shape elements fit in int32_t)");
        }
        out.push_back(static_cast<::mlx::core::ShapeElem>(d));
    }
    return out;
}

namespace {
// All shared_ptr<mlx::core::array> instances built by this bridge use the
// same deleter pattern: free the wrapped object and notify MemoryTracker.
// Centralizing here avoids drift between upload/wrap call sites.
std::shared_ptr<::mlx::core::array>
make_tracked(::mlx::core::array* raw, std::size_t bytes) {
    if (bytes > 0) {
        MemoryTracker::track_alloc(bytes, Device::GPU);
    }
    return std::shared_ptr<::mlx::core::array>(
        raw, [bytes](::mlx::core::array* p) {
            delete p;
            if (bytes > 0) {
                MemoryTracker::track_free(bytes, Device::GPU);
            }
        });
}
}  // namespace

GpuStorage upload_cpu_to_gpu(const CpuStorage& cpu, const Shape& shape) {
    auto mlx_shape = to_mlx_shape(shape);
    auto mlx_dt = to_mlx_dtype(cpu.dtype);

    // MLX takes (void*, Shape, Dtype, deleter). Capture the CpuStorage's
    // shared_ptr by value into the deleter so the source buffer outlives
    // the MLX array no matter how MLX uses it (zero-copy or internal copy).
    auto keepalive = std::make_shared<std::shared_ptr<std::byte[]>>(cpu.ptr);
    void* raw = static_cast<void*>(cpu.ptr.get());

    ::mlx::core::array external(
        raw, std::move(mlx_shape), mlx_dt,
        [keepalive](void* /*p*/) mutable { keepalive.reset(); });

    // Force a copy into MLX-owned memory. The void*-constructor builds an
    // "external" array whose data is the host buffer; downstream MLX ops on
    // that array (transpose, conv2d) can produce wrong results when the
    // graph reasons about strides assuming MLX-owned contiguous memory.
    // Copying eagerly via mlx::core::copy() materializes a fresh MLX-owned
    // array with canonical strides — at the cost of one host→device copy.
    auto owned = ::mlx::core::copy(external);

    GpuStorage out;
    out.dtype = cpu.dtype;
    out.nbytes = cpu.nbytes;
    out.arr = make_tracked(new ::mlx::core::array(std::move(owned)),
                           out.nbytes);
    return out;
}

CpuStorage download_gpu_to_cpu(const GpuStorage& gpu, const Shape& shape) {
    if (!gpu.arr) {
        throw LucidError("download_gpu_to_cpu: null GPU array");
    }
    // Force any pending compute graph to materialize before reading.
    gpu.arr->eval();

    const std::size_t total = shape_numel(shape) * dtype_size(gpu.dtype);
    CpuStorage out;
    out.dtype = gpu.dtype;
    out.nbytes = total;
    if (total == 0) {
        return out;
    }
    out.ptr = allocate_aligned_bytes(total, Device::CPU);

    // After eval(), MLX exposes a host-readable pointer via `data<T>()`. We
    // use the byte form (data<uint8_t>) and memcpy. MLX guarantees this is
    // contiguous in row-major after eval for arrays we constructed.
    const auto* src = gpu.arr->data<std::uint8_t>();
    std::memcpy(out.ptr.get(), src, total);
    return out;
}

GpuStorage wrap_mlx_array(::mlx::core::array&& arr, Dtype dtype) {
    GpuStorage out;
    out.dtype = dtype;
    out.nbytes = arr.nbytes();
    out.arr = make_tracked(new ::mlx::core::array(std::move(arr)), out.nbytes);
    return out;
}

}  // namespace lucid::gpu
