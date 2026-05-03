// lucid/_C/core/Storage.h
//
// Discriminated-union storage type that represents the three memory domains
// supported by the engine.  All three variants carry a shared VersionCounter
// so that autograd can detect in-place mutations between the forward and
// backward passes.
//
// CpuStorage  — heap memory allocated via posix_memalign (64-byte aligned),
//               owned through a shared_ptr<byte[]>.  Multiple TensorImpl
//               objects can share the same CpuStorage (views / slices).
//
// GpuStorage  — wraps an mlx::core::array, which is an opaque handle to a
//               node in the MLX lazy evaluation graph.  The underlying buffer
//               lives in GPU-private memory; direct CPU pointer access before
//               graph evaluation causes a SIGBUS.
//
// SharedStorage — a Metal buffer allocated with MTLResourceStorageModeShared,
//                making it simultaneously accessible from the CPU (cpu_ptr)
//                and the GPU (mtl_handle).  Useful for zero-copy host↔GPU
//                transfers on Apple Silicon unified memory hardware.  The
//                lifetime of the underlying allocation is managed through the
//                owner shared_ptr.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

#include "Dtype.h"

namespace mlx::core {
class array;
}

namespace lucid {

// Monotonically increasing counter incremented on every in-place write.
// Autograd saves the version at forward time and checks it at backward time
// to detect illegal mutations (see VersionMismatch in Error.h).
using VersionCounter = std::atomic<std::uint64_t>;

// Flat CPU buffer descriptor.  Shared between DataBuffer and CpuStorage so
// that the allocator's StoragePtr abstraction can be used interchangeably with
// the tensor-level CpuStorage without an extra copy.
struct DataBuffer {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    // Atomically increments the version counter.  Uses relaxed ordering
    // because version bumps are sequenced by the surrounding operation.
    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

// Alias used by the Allocator layer to return typed buffer descriptors.
using StoragePtr = std::shared_ptr<DataBuffer>;

// CPU tensor storage backed by a posix_memalign allocation.
//
// Ownership is shared: multiple TensorImpl views may hold a copy of ptr
// that points into the same physical allocation.  The VersionCounter is also
// shared across views so that a mutation through one view is visible to all.
// The to_data_buffer() helper round-trips the CpuStorage through a DataBuffer
// for callers that operate at the allocator abstraction level.
struct CpuStorage {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    CpuStorage() = default;

    CpuStorage(std::shared_ptr<std::byte[]> p, std::size_t nb, Dtype dt)
        : ptr(std::move(p)), nbytes(nb), dtype(dt) {}

    // Adopts the ptr, nbytes, dtype, and version from an existing DataBuffer,
    // allowing the allocator's StoragePtr to be injected directly.
    explicit CpuStorage(StoragePtr buf)
        : ptr(buf->ptr), nbytes(buf->nbytes), dtype(buf->dtype), version(buf->version) {}

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }

    // Wraps this CpuStorage back into a DataBuffer.  The returned DataBuffer
    // shares the same ptr and version as this object — no allocation occurs.
    StoragePtr to_data_buffer() const {
        auto buf = std::make_shared<DataBuffer>();
        buf->ptr = ptr;
        buf->nbytes = nbytes;
        buf->dtype = dtype;
        buf->version = version;
        return buf;
    }
};

// GPU tensor storage backed by an mlx::core::array lazy graph node.
//
// The arr shared_ptr keeps the MLX node alive as long as this GpuStorage
// (and any TensorImpl that holds it) is in scope.  Operations that need the
// GPU data must go through the MLX evaluation path — never cast arr's
// internal pointer to a CPU pointer.
struct GpuStorage {
    std::shared_ptr<mlx::core::array> arr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

// Metal shared-memory storage accessible from both CPU and GPU.
//
// cpu_ptr points into a Metal buffer with MTLResourceStorageModeShared, which
// is mapped into the process address space.  mtl_handle is the opaque
// id<MTLBuffer> pointer, kept as void* to avoid pulling in Metal headers here.
// The owner shared_ptr manages the lifetime of the underlying Metal allocation;
// cpu_view() creates a CpuStorage whose deleter holds a reference to owner so
// the Metal buffer is not freed while the CPU view is alive.
struct SharedStorage {
    void* cpu_ptr = nullptr;
    void* mtl_handle = nullptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);
    // Retains the underlying Metal allocation.
    std::shared_ptr<void> owner;

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }

    // Returns a CpuStorage that aliases cpu_ptr without copying data.
    // The returned storage's deleter holds a reference to owner, ensuring
    // the Metal buffer outlives the CpuStorage.
    CpuStorage cpu_view() const {
        auto tok = owner;
        auto ptr = std::shared_ptr<std::byte[]>(
            static_cast<std::byte*>(cpu_ptr),
            // Custom deleter: release the owner token when the last CpuStorage
            // view goes away, which allows the Metal buffer to be freed.
            [tok = std::move(tok)](std::byte*) mutable { tok.reset(); });

        CpuStorage cv(std::move(ptr), nbytes, dtype);
        cv.version = version;
        return cv;
    }
};

// The discriminated union type held by TensorImpl.  Variant index 0 = CPU,
// 1 = GPU, 2 = SharedStorage — the storage_is_* helpers below rely on this
// ordering and must be updated if the variant order changes.
using Storage = std::variant<CpuStorage, GpuStorage, SharedStorage>;

// ---------------------------------------------------------------------------
// Storage predicates and typed accessors
// ---------------------------------------------------------------------------

inline bool storage_is_cpu(const Storage& s) noexcept {
    return s.index() == 0;
}
inline bool storage_is_gpu(const Storage& s) noexcept {
    return s.index() == 1;
}

inline bool storage_is_metal_shared(const Storage& s) noexcept {
    return s.index() == 2;
}

// Returns the allocated byte size for any Storage variant.
inline std::size_t storage_nbytes(const Storage& s) noexcept {
    switch (s.index()) {
    case 0:
        return std::get<0>(s).nbytes;
    case 1:
        return std::get<1>(s).nbytes;
    case 2:
        return std::get<2>(s).nbytes;
    default:
        return 0;
    }
}

// Returns the element dtype for any Storage variant.
inline Dtype storage_dtype(const Storage& s) noexcept {
    switch (s.index()) {
    case 0:
        return std::get<0>(s).dtype;
    case 1:
        return std::get<1>(s).dtype;
    case 2:
        return std::get<2>(s).dtype;
    default:
        return Dtype::F32;
    }
}

// Typed const/mutable accessors — undefined behaviour if the variant holds a
// different alternative (std::get will throw std::bad_variant_access in debug
// builds).
inline const CpuStorage& storage_cpu(const Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}
inline CpuStorage& storage_cpu(Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}

inline const GpuStorage& storage_gpu(const Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}
inline GpuStorage& storage_gpu(Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}

inline const SharedStorage& storage_metal_shared(const Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}
inline SharedStorage& storage_metal_shared(Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}

}  // namespace lucid
