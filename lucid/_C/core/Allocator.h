#pragma once

#include <cstddef>
#include <memory>

#include "Device.h"

namespace lucid {

// All CpuStorage allocations are 64-byte aligned (Apple AMX & AVX-512 friendly).
constexpr std::size_t kCpuAlignment = 64;

// Allocate and register with the per-device MemoryTracker. The returned
// shared_ptr's deleter calls track_free, so usage is "fire and forget."
// Throws lucid::OutOfMemory on failure.
std::shared_ptr<std::byte[]> allocate_aligned_bytes(std::size_t nbytes,
                                                    Device device = Device::CPU);

}  // namespace lucid
