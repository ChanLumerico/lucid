// lucid/_C/core/MemoryStats.h
//
// Global allocation accounting for the Lucid engine.  MemoryTracker maintains
// separate atomic counters for CPU and GPU allocations so that diagnostic
// tools and Python-side memory queries can report live usage, peak usage, and
// allocation/free counts without acquiring any locks.
//
// MemoryTracker::track_alloc() and track_free() are called from the custom
// deleter installed by allocate_aligned_bytes() (Allocator.cpp) and should
// not be called by application code directly.
//
// Thread safety: all counter operations use std::atomic with
// memory_order_relaxed.  The counters are eventually consistent — a snapshot
// taken via get_stats() is not guaranteed to be a coherent view of a single
// instant, but the approximation is sufficient for diagnostic purposes.

#pragma once

#include <atomic>
#include <cstddef>

#include "../api.h"
#include "Device.h"

namespace lucid {

// Plain snapshot of allocation counters for one device, returned by
// MemoryTracker::get_stats().
struct LUCID_API MemoryStats {
    std::size_t current_bytes = 0;
    std::size_t peak_bytes = 0;
    std::size_t alloc_count = 0;
    std::size_t free_count = 0;
};

// Atomic allocation counter bank, one per device.
//
// All members are atomic so that track_alloc / track_free can be called from
// any thread without locking.  Peak tracking uses a compare-exchange loop to
// avoid a separate mutex while still achieving a monotonically non-decreasing
// peak counter.
class LUCID_API MemoryTracker {
public:
    struct Counters {
        std::atomic<std::size_t> current_bytes{0};
        std::atomic<std::size_t> peak_bytes{0};
        std::atomic<std::size_t> alloc_count{0};
        std::atomic<std::size_t> free_count{0};
    };

    // Records an allocation of nbytes on device.  Updates current_bytes,
    // alloc_count, and peak_bytes (if current_bytes now exceeds the previous
    // peak) using a CAS loop for the peak update.
    static void track_alloc(std::size_t nbytes, Device device);

    // Records a deallocation of nbytes on device.  Decrements current_bytes
    // and increments free_count.
    static void track_free(std::size_t nbytes, Device device);

    // Returns a snapshot of the counters for device.
    static MemoryStats get_stats(Device device);

    // Resets peak_bytes to the current current_bytes value, discarding the
    // historical maximum.  Useful at the start of a profiling window.
    static void reset_peak(Device device);

private:
    // Returns the Counters instance for the given device.
    static Counters& counters_for(Device device);
};

}  // namespace lucid
