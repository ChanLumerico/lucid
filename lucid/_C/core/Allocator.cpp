#include "Allocator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "Error.h"
#include "MemoryStats.h"

namespace lucid {

namespace {

constexpr std::size_t kMinClass = kCpuAlignment;
constexpr std::size_t kMaxClass = 4 * 1024 * 1024;
constexpr std::size_t kMaxDepth = 32;
constexpr std::size_t kNumClasses = 23;

inline std::size_t round_up_pool(std::size_t n) noexcept {
    std::size_t s = kMinClass;
    while (s < n)
        s <<= 1;
    return s;
}
inline int class_index(std::size_t rounded) noexcept {
    int idx = 0;
    std::size_t s = kMinClass;
    while (s < rounded && idx < static_cast<int>(kNumClasses) - 1) {
        s <<= 1;
        ++idx;
    }
    return idx;
}

struct ThreadPool {
    std::array<std::vector<void*>, kNumClasses> lists;

    void* pop(int cls) noexcept {
        auto& v = lists[static_cast<std::size_t>(cls)];
        if (v.empty())
            return nullptr;
        void* p = v.back();
        v.pop_back();
        return p;
    }
    bool push(int cls, void* p) noexcept {
        auto& v = lists[static_cast<std::size_t>(cls)];
        if (v.size() >= kMaxDepth)
            return false;
        v.push_back(p);
        return true;
    }
    ~ThreadPool() {
        for (auto& v : lists)
            for (void* p : v)
                std::free(p);
    }
};

thread_local ThreadPool t_pool;

}  // namespace

std::shared_ptr<std::byte[]> allocate_aligned_bytes(std::size_t nbytes, Device device) {
    if (nbytes == 0)
        return {};

    const bool poolable = (device == Device::CPU) && (nbytes <= kMaxClass);
    void* raw = nullptr;

    if (poolable) {
        const std::size_t rounded = round_up_pool(nbytes);
        const int cls = class_index(rounded);
        raw = t_pool.pop(cls);
        if (!raw) {
            if (::posix_memalign(&raw, kCpuAlignment, rounded) != 0 || !raw) {
                const auto s = MemoryTracker::get_stats(device);
                throw OutOfMemory(nbytes, s.current_bytes, s.peak_bytes,
                                  std::string(device_name(device)));
            }
        }
        MemoryTracker::track_alloc(nbytes, device);

        const std::size_t rounded_cap = rounded;
        auto deleter = [nbytes, device, rounded_cap](std::byte* p) {
            const int c = class_index(rounded_cap);
            if (!t_pool.push(c, p))
                std::free(p);
            MemoryTracker::track_free(nbytes, device);
        };
        return std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw), deleter);
    }

    if (::posix_memalign(&raw, kCpuAlignment, nbytes) != 0 || !raw) {
        const auto s = MemoryTracker::get_stats(device);
        throw OutOfMemory(nbytes, s.current_bytes, s.peak_bytes, std::string(device_name(device)));
    }
    MemoryTracker::track_alloc(nbytes, device);
    auto deleter = [nbytes, device](std::byte* p) {
        std::free(p);
        MemoryTracker::track_free(nbytes, device);
    };
    return std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw), deleter);
}

}  // namespace lucid
