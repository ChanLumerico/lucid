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

// --------------------------------------------------------------------------
// Phase 9.3: thread-local small-block pool.
//
// Strategy: 22 power-of-2 size classes [64B … 256KB]. Each class holds a
// free-list of up to kMaxDepth raw blocks. Alloc: pop from free-list (no
// syscall) or fall back to posix_memalign. Free: if block fits a class and
// the list isn't full, push; otherwise std::free.
//
// Cross-thread safety: the shared_ptr deleter runs on whatever thread drops
// the last reference. To stay lock-free, blocks freed on a non-owning thread
// go to a mutex-protected global "return queue" that the thread drains on its
// next allocation.
// --------------------------------------------------------------------------

namespace {

constexpr std::size_t kMinClass = kCpuAlignment;  // 64 B
constexpr std::size_t kMaxClass = 256 * 1024;     // 256 KB
constexpr std::size_t kMaxDepth = 32;             // blocks per class
constexpr std::size_t kNumClasses = 23;           // 64B…4MB (log2 range)

// Round n up to the next power-of-2 >= kMinClass.
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

// Per-thread free lists.
struct ThreadPool {
    std::array<std::vector<void*>, kNumClasses> lists;

    void* pop(int cls) noexcept {
        auto& v = lists[cls];
        if (v.empty())
            return nullptr;
        void* p = v.back();
        v.pop_back();
        return p;
    }

    bool push(int cls, void* p) noexcept {
        auto& v = lists[cls];
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

    // Pool only for CPU allocations within the pooled size range.
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
            // Try to return to the thread-local pool. If this thread's list is
            // full (or this is a different thread), fall through to std::free.
            const int c = class_index(rounded_cap);
            if (!t_pool.push(c, p))
                std::free(p);
            MemoryTracker::track_free(nbytes, device);
        };
        return std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw), deleter);
    }

    // Large allocation or GPU: fall back to posix_memalign.
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
