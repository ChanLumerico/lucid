// lucid/_C/backend/gpu/MetalAllocator.h
//
// Thin C++ interface over MTLDevice buffer allocation.  All buffers produced
// here use MTLResourceStorageModeShared, which means the same physical pages
// are accessible from both the CPU (via cpu_ptr) and from GPU compute kernels
// (via the MTLBuffer handle) without any copy.  This is only possible on Apple
// Silicon due to the unified DRAM architecture.
//
// Ownership model:
//   MetalBuffer is a plain non-owning struct; the caller is responsible for
//   calling deallocate_shared().  OwnedMetalBuffer pairs a MetalBuffer with a
//   shared_ptr whose custom deleter calls deallocate_shared() on last release,
//   enabling ref-counted GPU-CPU shared memory regions.
//
// Note on wrap_existing: uses newBufferWithBytesNoCopy which requires the
// memory to be page-aligned and at least page-size bytes.  Use this only for
// MTLResourceStorageModeShared buffers you already own; MLX GPU-private arrays
// are NOT page-aligned and must not be passed here.

#pragma once

#include <cstddef>
#include <memory>

namespace lucid::gpu {

// Plain non-owning descriptor for a Metal shared buffer.
// cpu_ptr: CPU-accessible virtual address.
// mtl_handle: opaque CFRetain'd id<MTLBuffer> cast to void*.
// nbytes: allocation size in bytes.
struct MetalBuffer {
    void* cpu_ptr;
    void* mtl_handle;
    std::size_t nbytes;
};

// Allocates a new MTLResourceStorageModeShared buffer of nbytes bytes.
// Returns a zero-filled MetalBuffer on failure.
MetalBuffer allocate_shared(std::size_t nbytes);

// Releases the CFRetain'd MTLBuffer held in buf.mtl_handle and zeroes all
// fields.  Safe to call on a zero-initialised MetalBuffer (no-op).
void deallocate_shared(MetalBuffer& buf) noexcept;

// Wraps an existing page-aligned CPU pointer in a MTLBuffer without copying.
// The caller must ensure cpu_ptr remains valid for the lifetime of the buffer.
MetalBuffer wrap_existing(void* cpu_ptr, std::size_t nbytes);

// Allocates a shared buffer and returns it bundled with a ref-counted owner
// whose destructor calls deallocate_shared.  Preferred for use in Storage
// objects where RAII lifetime management is needed.
struct OwnedMetalBuffer {
    MetalBuffer buf;
    std::shared_ptr<void> owner;
};
OwnedMetalBuffer make_metal_shared(std::size_t nbytes);

}  // namespace lucid::gpu
