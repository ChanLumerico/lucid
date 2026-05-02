#pragma once

#include <cstddef>
#include <memory>

namespace lucid::gpu {

struct MetalBuffer {
    void* cpu_ptr;
    void* mtl_handle;
    std::size_t nbytes;
};

MetalBuffer allocate_shared(std::size_t nbytes);

void deallocate_shared(MetalBuffer& buf) noexcept;

MetalBuffer wrap_existing(void* cpu_ptr, std::size_t nbytes);

struct OwnedMetalBuffer {
    MetalBuffer buf;
    std::shared_ptr<void> owner;
};
OwnedMetalBuffer make_metal_shared(std::size_t nbytes);

}  // namespace lucid::gpu
