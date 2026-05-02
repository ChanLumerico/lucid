#include "MemoryStats.h"

#include <algorithm>

namespace lucid {

namespace {

MemoryTracker::Counters g_cpu;
MemoryTracker::Counters g_gpu;

}  // namespace

MemoryTracker::Counters& MemoryTracker::counters_for(Device device) {
    return device == Device::CPU ? g_cpu : g_gpu;
}

void MemoryTracker::track_alloc(std::size_t nbytes, Device device) {
    auto& c = counters_for(device);
    const auto cur = c.current_bytes.fetch_add(nbytes, std::memory_order_relaxed) + nbytes;
    c.alloc_count.fetch_add(1, std::memory_order_relaxed);

    auto peak = c.peak_bytes.load(std::memory_order_relaxed);
    while (cur > peak &&
           !c.peak_bytes.compare_exchange_weak(peak, cur, std::memory_order_relaxed)) {
    }
}

void MemoryTracker::track_free(std::size_t nbytes, Device device) {
    auto& c = counters_for(device);
    c.current_bytes.fetch_sub(nbytes, std::memory_order_relaxed);
    c.free_count.fetch_add(1, std::memory_order_relaxed);
}

MemoryStats MemoryTracker::get_stats(Device device) {
    auto& c = counters_for(device);
    MemoryStats s;
    s.current_bytes = c.current_bytes.load(std::memory_order_relaxed);
    s.peak_bytes = c.peak_bytes.load(std::memory_order_relaxed);
    s.alloc_count = c.alloc_count.load(std::memory_order_relaxed);
    s.free_count = c.free_count.load(std::memory_order_relaxed);
    return s;
}

void MemoryTracker::reset_peak(Device device) {
    auto& c = counters_for(device);
    c.peak_bytes.store(c.current_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

}  // namespace lucid
