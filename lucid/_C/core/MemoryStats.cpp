// lucid/_C/core/MemoryStats.cpp
//
// Per-device allocation counter implementation.  Two static Counters objects
// (g_cpu, g_gpu) are stored in anonymous namespace to avoid external linkage.
// All operations use memory_order_relaxed because the counters serve only as
// approximate diagnostics — there is no happens-before dependency between a
// counter update and the compute work that follows it.

#include "MemoryStats.h"

#include <algorithm>

namespace lucid {

namespace {

// Separate counters for CPU and GPU allocations.
MemoryTracker::Counters g_cpu;
MemoryTracker::Counters g_gpu;

// Host reads of evaluated engine arrays: count and bytes.
std::atomic<std::size_t> g_host_syncs{0};
std::atomic<std::size_t> g_host_sync_bytes{0};

}  // namespace

void MemoryTracker::track_host_sync(std::size_t nbytes) {
    g_host_syncs.fetch_add(1, std::memory_order_relaxed);
    g_host_sync_bytes.fetch_add(nbytes, std::memory_order_relaxed);
}

std::size_t MemoryTracker::host_sync_count() {
    return g_host_syncs.load(std::memory_order_relaxed);
}

std::size_t MemoryTracker::host_sync_bytes() {
    return g_host_sync_bytes.load(std::memory_order_relaxed);
}

MemoryTracker::Counters& MemoryTracker::counters_for(Device device) {
    return device == Device::CPU ? g_cpu : g_gpu;
}

void MemoryTracker::track_alloc(std::size_t nbytes, Device device) {
    auto& c = counters_for(device);
    // fetch_add returns the previous value; add nbytes to get the new total.
    const auto cur = c.current_bytes.fetch_add(nbytes, std::memory_order_relaxed) + nbytes;
    c.alloc_count.fetch_add(1, std::memory_order_relaxed);

    // Update peak with a CAS loop: if cur > peak, attempt to write cur; retry
    // if another thread raced ahead.  The loop terminates when either our
    // write succeeds or another thread already stored a value >= cur.
    // compare_exchange_weak is preferred here over the _strong variant because
    // the weak form avoids a redundant load on platforms where CAS may
    // spuriously fail; the while-condition handles the spurious case.
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
    // Resetting peak to current is a single store; no CAS needed.
    c.peak_bytes.store(c.current_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

}  // namespace lucid
