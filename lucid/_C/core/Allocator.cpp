#include "Allocator.h"

#include <cstdlib>
#include <string>

#include "Exceptions.h"
#include "MemoryStats.h"

namespace lucid {

std::shared_ptr<std::byte[]> allocate_aligned_bytes(std::size_t nbytes, Device device) {
    if (nbytes == 0) {
        return {};
    }
    void* raw = nullptr;
    if (::posix_memalign(&raw, kCpuAlignment, nbytes) != 0 || raw == nullptr) {
        const auto stats = MemoryTracker::get_stats(device);
        throw OutOfMemory(nbytes, stats.current_bytes, stats.peak_bytes,
                          std::string(device_name(device)));
    }

    MemoryTracker::track_alloc(nbytes, device);

    // The deleter knows how many bytes to release because we capture nbytes
    // and the device by value into the closure. shared_ptr stores the deleter
    // type-erased, so this stays single-allocation under the hood.
    auto deleter = [nbytes, device](std::byte* p) {
        std::free(p);
        MemoryTracker::track_free(nbytes, device);
    };
    return std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw), deleter);
}

}  // namespace lucid
