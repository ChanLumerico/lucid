#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// Phase 2.5 — Storage ownership model
//
// VersionCounter: a shared, atomically-incremented counter attached to every
//   mutable storage buffer. Autograd saved-tensor checks compare the counter
//   at save-time vs use-time to detect in-place mutations. Shared across all
//   views/slices that alias the same physical buffer.
//
// DataBuffer: the named backing-store concept for CPU tensors. Bundles the
//   raw byte allocation, size, dtype, and version counter in one object.
//   Multiple TensorImpls may share a single DataBuffer (aliasing / views).
//
// StoragePtr: shared_ptr<DataBuffer> — canonical way to transfer ownership
//   of a CPU buffer between ops, backends, and autograd nodes.
//
// GpuStorage already satisfies equivalent semantics via MLX's built-in
// shared_ptr<array> (CoW, immutable-on-write). Its version field tracks
// in-place MLX mutations (currently always 0 for immutable compute graph).
// ---------------------------------------------------------------------------

using VersionCounter = std::atomic<std::uint64_t>;

struct DataBuffer {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    /// Increment version; call before any in-place mutation of `ptr`.
    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

using StoragePtr = std::shared_ptr<DataBuffer>;

// CpuStorage — owns or aliases a CPU byte buffer.
//
// The `.ptr` / `.nbytes` / `.dtype` fields are kept for backward compatibility
// with the 600+ existing call-sites. The `.version` field is a shared counter
// that is also exposed on the DataBuffer surface; callers that want view
// semantics should share a StoragePtr and construct CpuStorage from it.
struct CpuStorage {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    /// Construct from a DataBuffer (shares the version counter).
    CpuStorage() = default;

    /// Replicate old aggregate initialisation: CpuStorage{ptr, nbytes, dtype}.
    CpuStorage(std::shared_ptr<std::byte[]> p, std::size_t nb, Dtype dt)
        : ptr(std::move(p)), nbytes(nb), dtype(dt) {}

    /// Construct from a DataBuffer — shares the same allocation and version counter.
    explicit CpuStorage(StoragePtr buf)
        : ptr(buf->ptr), nbytes(buf->nbytes), dtype(buf->dtype), version(buf->version) {}

    /// Increment version before any in-place write to this buffer.
    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }

    /// Export as a DataBuffer (shares the same allocation and version).
    StoragePtr to_data_buffer() const {
        auto buf = std::make_shared<DataBuffer>();
        buf->ptr = ptr;
        buf->nbytes = nbytes;
        buf->dtype = dtype;
        buf->version = version;
        return buf;
    }
};

// Phase 3.7: GPU storage holds a shared_ptr to an mlx::core::array. MLX
// arrays are inherently shareable (CoW desc internally), so wrapping in
// shared_ptr keeps copy/move of Storage cheap and lets multiple TensorImpls
// observe the same MLX array without redundant allocations. The `dtype`
// and `nbytes` fields mirror CpuStorage so dispatch sites that only need
// metadata don't have to peek into the MLX array.
//
// Phase 2.5 addition: `version` tracks in-place MLX mutations (currently
// always 0 for immutable compute graph; incremented by mutable_storage()
// users that overwrite the arr pointer in-place).
struct GpuStorage {
    std::shared_ptr<mlx::core::array> arr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

// Phase 9.2 — SharedStorage: Apple Silicon unified-memory buffer.
//
// On Apple Silicon the CPU and GPU share the same physical DRAM. A
// MTLBuffer allocated with MTLResourceStorageModeShared gives a single
// region that is directly accessible from both sides without any memcpy.
//
// cpu_ptr   : the Metal buffer's `contents` pointer (CPU-readable/writable).
// mtl_handle: opaque id<MTLBuffer> retained by the MetalAllocator layer.
// owner     : shared_ptr<void> RAII guard; drop to release the Metal buffer.
//
// cpu_view() fabricates a lightweight CpuStorage view over cpu_ptr so that
// existing CPU-only paths (accumulate_into, numpy export, copy_from) can
// reuse SharedStorage data without extra copies.
struct SharedStorage {
    void*       cpu_ptr    = nullptr;
    void*       mtl_handle = nullptr;  ///< id<MTLBuffer> (retained via owner)
    std::size_t nbytes     = 0;
    Dtype       dtype      = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);
    std::shared_ptr<void>           owner;  ///< keeps Metal buffer alive

    void bump_version() const noexcept {
        version->fetch_add(1, std::memory_order_relaxed);
    }
    std::uint64_t get_version() const noexcept {
        return version->load(std::memory_order_relaxed);
    }

    /// Return a CpuStorage that aliases cpu_ptr.  The owner token ensures the
    /// Metal buffer stays alive while the returned CpuStorage is live.
    CpuStorage cpu_view() const {
        auto tok = owner;  // copy keeps Metal buffer alive for the view's lifetime
        auto ptr = std::shared_ptr<std::byte[]>(
            static_cast<std::byte*>(cpu_ptr),
            [tok = std::move(tok)](std::byte*) mutable { tok.reset(); });
        // CpuStorage 3-arg ctor: (ptr, nbytes, dtype).  Then share the same
        // VersionCounter so mutations via either handle are visible to the other.
        CpuStorage cv(std::move(ptr), nbytes, dtype);
        cv.version = version;
        return cv;
    }
};

// Three-way variant: CPU heap | MLX GPU array | Metal unified-memory buffer.
using Storage = std::variant<CpuStorage, GpuStorage, SharedStorage>;

// --------------------------------------------------------------------------- //
// Storage free-function helpers — avoid repeated std::visit boilerplate.
// --------------------------------------------------------------------------- //

inline bool storage_is_cpu(const Storage& s) noexcept {
    return s.index() == 0;
}
inline bool storage_is_gpu(const Storage& s) noexcept {
    return s.index() == 1;
}
/// True when the Storage wraps a Metal shared-memory buffer (Phase 9.2).
inline bool storage_is_metal_shared(const Storage& s) noexcept {
    return s.index() == 2;
}
inline std::size_t storage_nbytes(const Storage& s) noexcept {
    switch (s.index()) {
        case 0: return std::get<0>(s).nbytes;
        case 1: return std::get<1>(s).nbytes;
        case 2: return std::get<2>(s).nbytes;
        default: return 0;
    }
}
inline Dtype storage_dtype(const Storage& s) noexcept {
    switch (s.index()) {
        case 0: return std::get<0>(s).dtype;
        case 1: return std::get<1>(s).dtype;
        case 2: return std::get<2>(s).dtype;
        default: return Dtype::F32;
    }
}

// --------------------------------------------------------------------------- //
// Typed accessors — preferred over raw std::get<> at call sites.
// These make the intent explicit and allow future storage-type refactoring
// without touching every call site.
// --------------------------------------------------------------------------- //

/// Return the CpuStorage inside s.  Undefined behaviour if s holds GpuStorage.
inline const CpuStorage& storage_cpu(const Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}
inline CpuStorage& storage_cpu(Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}

/// Return the GpuStorage inside s.  Undefined behaviour if s holds CpuStorage.
inline const GpuStorage& storage_gpu(const Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}
inline GpuStorage& storage_gpu(Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}

/// Return the SharedStorage inside s.  Undefined behaviour otherwise.
inline const SharedStorage& storage_metal_shared(const Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}
inline SharedStorage& storage_metal_shared(Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}

}  // namespace lucid
