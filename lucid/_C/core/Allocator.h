#pragma once

#include <cstddef>
#include <memory>

#include "Device.h"

namespace lucid {

constexpr std::size_t kCpuAlignment = 64;

std::shared_ptr<std::byte[]> allocate_aligned_bytes(std::size_t nbytes,
                                                    Device device = Device::CPU);

}  // namespace lucid
