// lucid/_C/core/Storage.h
//
// Discriminated-union storage type representing the three memory domains the
// engine can own: CPU heap (Apple Accelerate), GPU lazy graph (MLX), and
// Metal shared-mode unified-memory buffers (host + device aliased).
//
// All three variants carry a shared :type:`VersionCounter` so that autograd
// can detect in-place mutations between forward and backward passes — see
// :class:`AutogradMeta::version` in :file:`TensorMeta.h`.
//
// Variant summary
// ---------------
// :class:`CpuStorage`
//     ``posix_memalign``-allocated 64-byte-aligned host buffer owned through
//     a ``shared_ptr<byte[]>``.  Multiple :class:`TensorImpl` views can share
//     the same underlying allocation.
//
// :class:`GpuStorage`
//     Wraps an :class:`mlx::core::array` node in the MLX lazy graph.  The
//     backing buffer is GPU-private; the CPU pointer is **not** dereferenceable
//     before :func:`TensorImpl::eval`.
//
// :class:`SharedStorage`
//     Metal buffer allocated with ``MTLResourceStorageModeShared``, mapped
//     into the process address space so the CPU and GPU can both touch it
//     without explicit copies.  Used for zero-copy host↔device transfers on
//     Apple Silicon unified-memory hardware.
//
// See Also
// --------
// :class:`TensorImpl` — owner of the :type:`Storage` variant.
// :class:`Dtype`      — element type recorded inside every variant.

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

// Monotonically increasing counter incremented on every in-place write to a
// storage buffer.
//
// Autograd snapshots the counter value at forward time and compares it at
// backward time — any mismatch raises :class:`VersionMismatch` (see
// :file:`Error.h`), which detects illegal mutations of saved tensors.  The
// counter is shared across all views of the same allocation via a
// ``shared_ptr``, so a mutation through any view is observable from every
// view.
using VersionCounter = std::atomic<std::uint64_t>;

// Flat untyped CPU buffer descriptor produced by the :class:`Allocator`.
//
// Shared with :class:`CpuStorage` so that the allocator's :type:`StoragePtr`
// abstraction can be threaded through tensor construction without an extra
// copy or wrapper layer.  Treat as an allocator-layer view of the same
// underlying memory; :class:`CpuStorage` is the tensor-layer wrapper.
//
// Attributes
// ----------
// ptr : std::shared_ptr<std::byte[]>
//     Owning pointer to the byte buffer.  Shared between every descriptor
//     and tensor that references the same allocation.
// nbytes : std::size_t
//     Allocated capacity in bytes.
// dtype : Dtype
//     Element type stored in the buffer.  Default :attr:`Dtype::F32`.
// version : std::shared_ptr<VersionCounter>
//     Atomic mutation counter shared with every alias of this allocation.
//
// See Also
// --------
// :class:`CpuStorage` — tensor-layer wrapper around the same bytes.
struct DataBuffer {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    // Atomically increments the version counter.
    //
    // Uses ``memory_order_relaxed`` because version bumps are sequenced by
    // the surrounding op — autograd only inspects the counter at well-defined
    // synchronisation points (forward save / backward replay).
    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }

    // Atomically reads the current version counter value.
    //
    // Returns
    // -------
    // std::uint64_t
    //     Monotonically non-decreasing snapshot of mutation count.
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

// Shared-ownership handle to a :class:`DataBuffer`.
//
// The :class:`Allocator` layer returns instances of this type so that the
// tensor layer can adopt the buffer without copying — see
// :class:`CpuStorage`'s :type:`StoragePtr` constructor.
using StoragePtr = std::shared_ptr<DataBuffer>;

// CPU tensor storage backed by a ``posix_memalign`` allocation.
//
// Ownership is shared: multiple :class:`TensorImpl` views may hold a copy of
// ``ptr`` that points into the same physical allocation.  The
// :type:`VersionCounter` is also shared so a mutation through one view is
// visible to every other view that aliases the same bytes.
//
// All Apple Accelerate / BLAS / vDSP code paths read and write this variant.
//
// Attributes
// ----------
// ptr : std::shared_ptr<std::byte[]>
//     64-byte aligned heap allocation, shared with every aliasing view.
// nbytes : std::size_t
//     Allocated capacity in bytes.
// dtype : Dtype
//     Element type.  Default :attr:`Dtype::F32`.
// version : std::shared_ptr<VersionCounter>
//     Shared mutation counter; in-place ops call :func:`bump_version`.
//
// See Also
// --------
// :class:`DataBuffer`     — allocator-layer twin.
// :class:`GpuStorage`     — GPU sibling variant.
// :class:`SharedStorage`  — unified-memory sibling variant.
struct CpuStorage {
    std::shared_ptr<std::byte[]> ptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    // Default-constructs an empty :class:`CpuStorage` (null pointer, zero
    // bytes, default F32 dtype, fresh version counter).
    CpuStorage() = default;

    // Constructs a :class:`CpuStorage` from raw pieces.
    //
    // Parameters
    // ----------
    // p : std::shared_ptr<std::byte[]>
    //     Allocation to adopt.
    // nb : std::size_t
    //     Allocation capacity in bytes.
    // dt : Dtype
    //     Element type stored in the buffer.
    CpuStorage(std::shared_ptr<std::byte[]> p, std::size_t nb, Dtype dt)
        : ptr(std::move(p)), nbytes(nb), dtype(dt) {}

    // Adopts the pointer, capacity, dtype, and version counter from an
    // existing :class:`DataBuffer`.
    //
    // Parameters
    // ----------
    // buf : StoragePtr
    //     Allocator-layer buffer to wrap.  Must be non-null.
    //
    // Notes
    // -----
    // The two objects share the same underlying allocation **and** version
    // counter after construction — bumping the version on either is visible
    // on both.
    explicit CpuStorage(StoragePtr buf)
        : ptr(buf->ptr), nbytes(buf->nbytes), dtype(buf->dtype), version(buf->version) {}

    // Atomically increments the shared version counter.
    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }

    // Reads the current version counter value.
    //
    // Returns
    // -------
    // std::uint64_t
    //     Monotonically non-decreasing snapshot of mutation count.
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }

    // Wraps this :class:`CpuStorage` back into a :class:`DataBuffer`.
    //
    // No allocation occurs — the returned :type:`StoragePtr` aliases the
    // same ``ptr`` and shares the same ``version`` counter.
    //
    // Returns
    // -------
    // StoragePtr
    //     Heap-allocated :class:`DataBuffer` mirroring this storage's
    //     fields.
    StoragePtr to_data_buffer() const {
        auto buf = std::make_shared<DataBuffer>();
        buf->ptr = ptr;
        buf->nbytes = nbytes;
        buf->dtype = dtype;
        buf->version = version;
        return buf;
    }
};

// GPU tensor storage backed by an :class:`mlx::core::array` lazy graph node.
//
// The ``arr`` shared pointer keeps the MLX graph node alive for as long as
// this :class:`GpuStorage` (and any :class:`TensorImpl` that holds it) is in
// scope.  All GPU-stream operations dispatch through MLX; **never** cast the
// MLX array's internal pointer to a CPU pointer — the buffer is GPU-private
// until evaluated (see :func:`TensorImpl::eval`).
//
// Attributes
// ----------
// arr : std::shared_ptr<mlx::core::array>
//     Opaque handle to the MLX lazy graph node.
// nbytes : std::size_t
//     Allocation capacity in bytes (logical — MLX may not have materialised
//     the buffer yet).
// dtype : Dtype
//     Element type.  Default :attr:`Dtype::F32`.
// version : std::shared_ptr<VersionCounter>
//     Shared mutation counter — same semantics as :class:`CpuStorage`.
//
// See Also
// --------
// :class:`CpuStorage`     — CPU sibling variant.
// :class:`SharedStorage`  — unified-memory sibling variant.
struct GpuStorage {
    std::shared_ptr<mlx::core::array> arr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);

    // Atomically increments the shared version counter.
    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }

    // Reads the current version counter value.
    //
    // Returns
    // -------
    // std::uint64_t
    //     Monotonically non-decreasing snapshot of mutation count.
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }
};

// Metal shared-memory storage accessible from both CPU and GPU.
//
// The underlying allocation is a Metal buffer created with
// ``MTLResourceStorageModeShared``, which the Metal driver maps into the
// process address space.  This gives Lucid a zero-copy host↔device bridge on
// Apple Silicon unified-memory hardware: the same bytes are addressable from
// CPU code through ``cpu_ptr`` and submitted to the GPU through
// ``mtl_handle``.
//
// Attributes
// ----------
// cpu_ptr : void*
//     Host-side mapped pointer.  Safe to dereference from any CPU thread.
// mtl_handle : void*
//     Opaque ``id<MTLBuffer>`` handle, stored as ``void*`` to avoid pulling
//     the Metal headers into this translation unit.
// nbytes : std::size_t
//     Allocation capacity in bytes.
// dtype : Dtype
//     Element type.  Default :attr:`Dtype::F32`.
// version : std::shared_ptr<VersionCounter>
//     Shared mutation counter — same semantics as the other variants.
// owner : std::shared_ptr<void>
//     Lifetime token for the underlying Metal allocation.  The
//     :func:`cpu_view` deleter captures a copy so the Metal buffer cannot
//     be freed while a CPU view is alive.
//
// See Also
// --------
// :func:`cpu_view` — produces a :class:`CpuStorage` aliasing this buffer.
struct SharedStorage {
    void* cpu_ptr = nullptr;
    void* mtl_handle = nullptr;
    std::size_t nbytes = 0;
    Dtype dtype = Dtype::F32;
    std::shared_ptr<VersionCounter> version = std::make_shared<VersionCounter>(0);
    // Retains the underlying Metal allocation.
    std::shared_ptr<void> owner;

    // Atomically increments the shared version counter.
    void bump_version() const noexcept { version->fetch_add(1, std::memory_order_relaxed); }

    // Reads the current version counter value.
    //
    // Returns
    // -------
    // std::uint64_t
    //     Monotonically non-decreasing snapshot of mutation count.
    std::uint64_t get_version() const noexcept { return version->load(std::memory_order_relaxed); }

    // Returns a :class:`CpuStorage` that aliases ``cpu_ptr`` without copying
    // bytes.
    //
    // The returned storage installs a custom deleter that holds a copy of
    // :attr:`owner`, guaranteeing the Metal buffer outlives every
    // :class:`CpuStorage` view spawned this way.  The :type:`VersionCounter`
    // is also shared with this :class:`SharedStorage`, so mutations through
    // the CPU view are visible to the GPU side.
    //
    // Returns
    // -------
    // CpuStorage
    //     CPU view aliasing the shared buffer.  Lifetime-tied to ``owner``.
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

// Discriminated union of the three storage variants, held by
// :class:`TensorImpl`.
//
// The variant index is part of the on-disk and runtime ABI: ``0`` =
// :class:`CpuStorage`, ``1`` = :class:`GpuStorage`, ``2`` =
// :class:`SharedStorage`.  The :func:`storage_is_cpu`, :func:`storage_is_gpu`,
// and :func:`storage_is_metal_shared` predicates rely on this ordering and
// must be updated in lockstep if the variant order ever changes.
using Storage = std::variant<CpuStorage, GpuStorage, SharedStorage>;

// ---------------------------------------------------------------------------
// Storage predicates and typed accessors
// ---------------------------------------------------------------------------

// Predicate: is ``s`` a :class:`CpuStorage`?
//
// Parameters
// ----------
// s : Storage
//     Variant to inspect.
//
// Returns
// -------
// bool
//     ``true`` iff the variant currently holds the :class:`CpuStorage`
//     alternative (index 0).
inline bool storage_is_cpu(const Storage& s) noexcept {
    return s.index() == 0;
}

// Predicate: is ``s`` a :class:`GpuStorage`?
//
// Parameters
// ----------
// s : Storage
//     Variant to inspect.
//
// Returns
// -------
// bool
//     ``true`` iff the variant currently holds the :class:`GpuStorage`
//     alternative (index 1).
inline bool storage_is_gpu(const Storage& s) noexcept {
    return s.index() == 1;
}

// Predicate: is ``s`` a :class:`SharedStorage`?
//
// Parameters
// ----------
// s : Storage
//     Variant to inspect.
//
// Returns
// -------
// bool
//     ``true`` iff the variant currently holds the :class:`SharedStorage`
//     alternative (index 2).
inline bool storage_is_metal_shared(const Storage& s) noexcept {
    return s.index() == 2;
}

// Returns the allocated byte size of any :class:`Storage` variant.
//
// Parameters
// ----------
// s : Storage
//     Variant to query.
//
// Returns
// -------
// std::size_t
//     The ``nbytes`` field of whichever alternative ``s`` holds.  ``0`` if
//     the variant index is unrecognised (cannot happen with the current
//     definition).
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

// Returns the element dtype recorded inside any :class:`Storage` variant.
//
// Parameters
// ----------
// s : Storage
//     Variant to query.
//
// Returns
// -------
// Dtype
//     The ``dtype`` field of whichever alternative ``s`` holds.  Falls back
//     to :attr:`Dtype::F32` for unrecognised variant indices (cannot happen
//     with the current definition).
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

// Returns a const reference to ``s`` typed as :class:`CpuStorage`.
//
// Parameters
// ----------
// s : Storage
//     Variant known to hold :class:`CpuStorage`.
//
// Returns
// -------
// const CpuStorage&
//     Reference into ``s``.
//
// Raises
// ------
// std::bad_variant_access
//     The variant does not currently hold :class:`CpuStorage` (debug
//     builds; release builds may invoke undefined behaviour).
inline const CpuStorage& storage_cpu(const Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}

// Mutable overload of :func:`storage_cpu`.
//
// Parameters
// ----------
// s : Storage
//     Variant known to hold :class:`CpuStorage`.
//
// Returns
// -------
// CpuStorage&
//     Mutable reference into ``s``.
inline CpuStorage& storage_cpu(Storage& s) noexcept {
    return std::get<CpuStorage>(s);
}

// Returns a const reference to ``s`` typed as :class:`GpuStorage`.
//
// Parameters
// ----------
// s : Storage
//     Variant known to hold :class:`GpuStorage`.
//
// Returns
// -------
// const GpuStorage&
//     Reference into ``s``.
//
// Raises
// ------
// std::bad_variant_access
//     The variant does not currently hold :class:`GpuStorage`.
inline const GpuStorage& storage_gpu(const Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}

// Mutable overload of :func:`storage_gpu`.
//
// Parameters
// ----------
// s : Storage
//     Variant known to hold :class:`GpuStorage`.
//
// Returns
// -------
// GpuStorage&
//     Mutable reference into ``s``.
inline GpuStorage& storage_gpu(Storage& s) noexcept {
    return std::get<GpuStorage>(s);
}

// Returns a const reference to ``s`` typed as :class:`SharedStorage`.
//
// Parameters
// ----------
// s : Storage
//     Variant known to hold :class:`SharedStorage`.
//
// Returns
// -------
// const SharedStorage&
//     Reference into ``s``.
//
// Raises
// ------
// std::bad_variant_access
//     The variant does not currently hold :class:`SharedStorage`.
inline const SharedStorage& storage_metal_shared(const Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}

// Mutable overload of :func:`storage_metal_shared`.
//
// Parameters
// ----------
// s : Storage
//     Variant known to hold :class:`SharedStorage`.
//
// Returns
// -------
// SharedStorage&
//     Mutable reference into ``s``.
inline SharedStorage& storage_metal_shared(Storage& s) noexcept {
    return std::get<SharedStorage>(s);
}

}  // namespace lucid
