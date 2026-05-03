// lucid/_C/core/Allocator.h
//
// Public interface for Lucid's CPU memory allocator.  All tensor data
// allocated for the CPU stream goes through allocate_aligned_bytes rather
// than new / malloc, for two reasons:
//
//   1. Alignment: Accelerate BLAS and vDSP routines require 16- or 64-byte
//      aligned buffers for maximum throughput.  kCpuAlignment = 64 satisfies
//      both requirements and matches typical ARM64 cache-line size.
//
//   2. Pooling: The implementation (Allocator.cpp) maintains a thread-local
//      slab pool for allocations up to 4 MB.  Returning a buffer to the pool
//      avoids the kernel round-trip of posix_memalign / free, which is
//      significant for short-lived intermediate tensors.
//
// MLX allocations (GpuStorage) are managed entirely by the MLX runtime and
// must not go through this allocator — they live in GPU-private memory and
// would cause a SIGBUS if accessed via a CPU pointer.

#pragma once

#include <cstddef>
#include <memory>

#include "Device.h"

namespace lucid {

// Required alignment for all CPU tensor buffers, in bytes.
// 64 bytes = one typical ARM64 / x86-64 cache line, and satisfies the
// minimum alignment required by Accelerate vDSP and BLAS routines.
constexpr std::size_t kCpuAlignment = 64;

// Allocates nbytes of kCpuAlignment-aligned memory and returns ownership as a
// shared_ptr<byte[]> whose deleter either returns the block to the thread-local
// pool (for small allocations on CPU) or calls std::free directly (for large
// allocations and GPU device).
//
// Throws OutOfMemory if posix_memalign fails.  Returns an empty shared_ptr for
// nbytes == 0.
//
// The device parameter controls which MemoryTracker counter is updated and
// whether the pool path is used (pooling is only enabled for Device::CPU).
std::shared_ptr<std::byte[]> allocate_aligned_bytes(std::size_t nbytes,
                                                    Device device = Device::CPU);

}  // namespace lucid
