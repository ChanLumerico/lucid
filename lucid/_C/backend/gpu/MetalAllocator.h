#pragma once

// =====================================================================
// Lucid C++ engine — MetalAllocator (Phase 9.1)
// =====================================================================
//
// Apple Silicon CPU and GPU share the same physical memory. Allocating a
// MTLBuffer with MTLResourceStorageModeShared gives a single contiguous
// region that is directly addressable from both the CPU (via `contents`)
// and any GPU compute kernel (via the MTLBuffer handle) — without any
// host-to-device copy.
//
// This header exposes three C++-visible entry points that are implemented
// in MetalAllocator.mm (Objective-C++) to avoid pulling Metal headers into
// every translation unit.
//
// Layer: backend/gpu/. Only included from .cpp files and from
//   backend/gpu/MetalAllocator.mm. Never include from public headers.

#include <cstddef>
#include <memory>

namespace lucid::gpu {

// ---- MetalBuffer -----------------------------------------------------------
//
// Plain-old-data bundle returned by the allocator. The caller is responsible
// for calling deallocate_shared() exactly once when it no longer needs the
// buffer (or wrapping in a shared_ptr with a deleter; see make_metal_shared).
//
struct MetalBuffer {
    void*       cpu_ptr;    ///< CPU-accessible base address (non-null on success)
    void*       mtl_handle; ///< Opaque id<MTLBuffer> (ARC/retain already applied)
    std::size_t nbytes;     ///< Allocation size in bytes
};

// Allocate `nbytes` bytes of Metal shared memory via:
//   [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared]
//
// Returns a MetalBuffer with cpu_ptr == MTLBuffer.contents and mtl_handle
// set to the retained id<MTLBuffer>. Returns {nullptr, nullptr, 0} on failure.
MetalBuffer allocate_shared(std::size_t nbytes);

// Release a Metal buffer acquired via allocate_shared or wrap_existing.
// Calls objc_release on mtl_handle and zeroes the struct.
void deallocate_shared(MetalBuffer& buf) noexcept;

// Wrap a pre-existing CPU pointer that is already page-aligned in a
// Metal buffer via:
//   [device newBufferWithBytesNoCopy:cpu_ptr length:nbytes
//            options:MTLResourceStorageModeShared deallocator:nil]
//
// The caller retains ownership of the underlying memory; the Metal buffer
// is merely a view. Call deallocate_shared() to release the Metal handle
// when done, but the original `cpu_ptr` allocation must be freed separately.
MetalBuffer wrap_existing(void* cpu_ptr, std::size_t nbytes);

// Convenience: allocate_shared + wrap in a shared_ptr<void> RAII guard.
// The deleter calls deallocate_shared when the refcount drops to zero.
// On allocation failure returns {nullptr, nullptr, 0} with an empty owner.
struct OwnedMetalBuffer {
    MetalBuffer       buf;
    std::shared_ptr<void> owner; ///< non-null iff buf.cpu_ptr != nullptr
};
OwnedMetalBuffer make_metal_shared(std::size_t nbytes);

}  // namespace lucid::gpu
