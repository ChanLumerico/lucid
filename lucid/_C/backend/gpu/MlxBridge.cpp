// lucid/_C/backend/gpu/MlxBridge.cpp
//
// Implements the Storage ↔ mlx::core::array bridge functions declared in
// MlxBridge.h.
//
// upload_cpu_to_gpu design:
//   A transient mlx::core::array is constructed that points at the CPU buffer
//   via a non-owning "external" view.  A shared_ptr keepalive token holds the
//   original CpuStorage ptr so it cannot be freed until after mlx::core::copy()
//   has finished reading it.  copy() returns a new array backed by GPU-private
//   memory that is safe to use from Metal compute kernels.
//
// download_gpu_to_cpu design:
//   Calls arr->eval() to force lazy graph execution and materialise the array,
//   then accesses data<uint8_t>() for a memcpy to a fresh aligned CPU buffer.
//   After eval() the data pointer is guaranteed CPU-accessible on Apple Silicon
//   unified-memory hardware only when the array is not on a pure GPU stream.
//
// make_tracked:
//   Wraps a raw mlx::core::array* in a shared_ptr with a custom deleter that
//   notifies MemoryTracker on deallocation.  All GpuStorage objects are created
//   via this helper so memory accounting is consistent.

#include "MlxBridge.h"

#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include <mlx/allocator.h>
#include <mlx/ops.h>

#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/MemoryStats.h"
#include "MetalAllocator.h"

namespace lucid::gpu {

::mlx::core::Dtype to_mlx_dtype(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return ::mlx::core::bool_;
    case Dtype::I8:
        return ::mlx::core::int8;
    case Dtype::I16:
        return ::mlx::core::int16;
    case Dtype::I32:
        return ::mlx::core::int32;
    case Dtype::I64:
        return ::mlx::core::int64;
    case Dtype::F16:
        return ::mlx::core::float16;
    case Dtype::F32:
        return ::mlx::core::float32;
    case Dtype::F64:

        ErrorBuilder("to_mlx_dtype")
            .not_implemented("float64 is not supported on GPU (MLX-Metal limitation). "
                             "Cast to float32 first, or keep the tensor on CPU.");
    case Dtype::C64:
        return ::mlx::core::complex64;
    }
    ErrorBuilder("to_mlx_dtype").fail("unknown Dtype");
}

Dtype from_mlx_dtype(::mlx::core::Dtype dt) {
    using V = ::mlx::core::Dtype::Val;
    switch (dt.val()) {
    case V::bool_:
        return Dtype::Bool;
    case V::int8:
        return Dtype::I8;
    case V::int16:
        return Dtype::I16;
    case V::int32:
        return Dtype::I32;
    case V::int64:
        return Dtype::I64;
    case V::float16:
        return Dtype::F16;
    case V::float32:
        return Dtype::F32;
    case V::float64:
        return Dtype::F64;
    case V::complex64:
        return Dtype::C64;
    default:
        ErrorBuilder("from_mlx_dtype")
            .not_implemented("unsupported MLX dtype (uint*/bfloat16 not yet wired into Lucid)");
    }
}

::mlx::core::Shape to_mlx_shape(const Shape& shape) {
    ::mlx::core::Shape out;
    out.reserve(shape.size());
    for (auto d : shape) {
        if (d > std::numeric_limits<::mlx::core::ShapeElem>::max()) {
            ErrorBuilder("to_mlx_shape")
                .fail("dimension exceeds int32 range (MLX requires shape elements fit in int32_t)");
        }
        out.push_back(static_cast<::mlx::core::ShapeElem>(d));
    }
    return out;
}

namespace {

// Creates a ref-counted mlx::core::array pointer that notifies MemoryTracker
// when the last reference is dropped.  bytes == 0 disables tracking (used for
// empty arrays returned from shape-only operations).
std::shared_ptr<::mlx::core::array> make_tracked(::mlx::core::array* raw, std::size_t bytes) {
    if (bytes > 0) {
        MemoryTracker::track_alloc(bytes, Device::GPU);
    }
    return std::shared_ptr<::mlx::core::array>(raw, [bytes](::mlx::core::array* p) {
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

    auto keepalive = std::make_shared<std::shared_ptr<std::byte[]>>(cpu.ptr);
    void* raw = static_cast<void*>(cpu.ptr.get());
    ::mlx::core::array external(raw, std::move(mlx_shape), mlx_dt,
                                [keepalive](void*) mutable { keepalive.reset(); });

    auto owned = ::mlx::core::copy(external);

    GpuStorage out;
    out.dtype = cpu.dtype;
    out.nbytes = cpu.nbytes;
    out.arr = make_tracked(new ::mlx::core::array(std::move(owned)), out.nbytes);
    return out;
}

CpuStorage download_gpu_to_cpu(const GpuStorage& gpu, const Shape& shape) {
    if (!gpu.arr) {
        ErrorBuilder("download_gpu_to_cpu").fail("null GPU array");
    }

    gpu.arr->eval();

    const std::size_t total = shape_numel(shape) * dtype_size(gpu.dtype);
    CpuStorage out;
    out.dtype = gpu.dtype;
    out.nbytes = total;
    if (total == 0) {
        return out;
    }
    out.ptr = allocate_aligned_bytes(total, Device::CPU);

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

GpuStorage shared_storage_to_gpu(const SharedStorage& sh, const Shape& shape) {
    if (!sh.cpu_ptr || sh.nbytes == 0)
        ErrorBuilder("shared_storage_to_gpu").fail("SharedStorage is empty");

    auto mlx_shape = to_mlx_shape(shape);
    auto mlx_dt = to_mlx_dtype(sh.dtype);

    auto owner_tok = sh.owner;
    ::mlx::core::array external(
        sh.cpu_ptr, std::move(mlx_shape), mlx_dt,
        [owner_tok = std::move(owner_tok)](void*) mutable { owner_tok.reset(); });

    GpuStorage out;
    out.dtype = sh.dtype;
    out.nbytes = sh.nbytes;
    out.arr = make_tracked(new ::mlx::core::array(std::move(external)), out.nbytes);
    return out;
}

}  // namespace lucid::gpu
