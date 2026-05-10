// lucid/_C/test/core/test_concurrency.cpp
//
// Concurrency stress tests for the engine's thread-safety primitives.
//
// Verifies that:
//   1. The CPU allocator can be hammered from multiple threads with no
//      crashes / leaks (thread-local pool design).
//   2. MemoryTracker counters remain consistent under heavy concurrent
//      allocate/free traffic (atomic CAS-loop peak tracking).
//   3. Generator's mutex serializes Philox state updates correctly when
//      shared across threads.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Device.h"
#include "../../core/Generator.h"
#include "../../core/MemoryStats.h"

using namespace lucid;

namespace {

constexpr std::size_t kAllocsPerThread = 1000;
constexpr int         kThreadCount     = 8;

}  // namespace

// 1. Thread-local allocator pool: hammer allocate_aligned_bytes from every
//    thread.  Pool reuse should make most allocs zero-syscall; correctness
//    requires that buffers from one thread are never returned to another's
//    pool (segfault / double-free would surface here under TSan/ASan).
TEST(Concurrency, AllocatorMultiThreadedHammer) {
    std::atomic<std::size_t> alloc_total{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (int t = 0; t < kThreadCount; ++t) {
        threads.emplace_back([&alloc_total, t]() {
            for (std::size_t i = 0; i < kAllocsPerThread; ++i) {
                // Vary the size to exercise multiple slab classes.
                const std::size_t nbytes = 64 + (i % 16) * 1024;
                auto buf = allocate_aligned_bytes(nbytes, Device::CPU);
                ASSERT_NE(buf.get(), nullptr);
                // Touch the memory so ASan/UBSan can flag stale pages.
                buf.get()[0]            = static_cast<std::byte>(i & 0xFF);
                buf.get()[nbytes - 1]   = static_cast<std::byte>(t & 0xFF);
                alloc_total.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(alloc_total.load(),
              static_cast<std::size_t>(kThreadCount) * kAllocsPerThread);
}

// 2. MemoryTracker counter consistency.  After all threads finish, the
//    delta between alloc_count and free_count should match outstanding
//    allocations (= 0 here because all buffers go out of scope).
TEST(Concurrency, MemoryTrackerCountersStable) {
    const MemoryStats before = MemoryTracker::get_stats(Device::CPU);

    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int t = 0; t < kThreadCount; ++t) {
        threads.emplace_back([]() {
            for (std::size_t i = 0; i < kAllocsPerThread; ++i) {
                auto buf = allocate_aligned_bytes(4096, Device::CPU);
                (void)buf;  // immediately freed at scope exit
            }
        });
    }
    for (auto& th : threads) th.join();

    const MemoryStats after = MemoryTracker::get_stats(Device::CPU);

    // alloc_count and free_count both bumped by exactly the same number.
    const std::size_t allocs = after.alloc_count - before.alloc_count;
    const std::size_t frees  = after.free_count  - before.free_count;
    EXPECT_EQ(allocs, frees);
    // current_bytes back to where it started (within process — no leaks here).
    EXPECT_EQ(after.current_bytes, before.current_bytes);
}

// 3. Generator mutex correctness — a shared Generator should serialize
//    Philox state updates.  Without the mutex, parallel sample() calls
//    would race the underlying counter and produce duplicate streams.
TEST(Concurrency, GeneratorMutexSerializesUpdates) {
    Generator gen(/*seed=*/12345ULL);

    std::vector<std::uint64_t> all_samples;
    all_samples.reserve(static_cast<std::size_t>(kThreadCount) * 100);
    std::mutex collect_mu;

    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int t = 0; t < kThreadCount; ++t) {
        threads.emplace_back([&]() {
            std::vector<std::uint64_t> local;
            local.reserve(100);
            for (int i = 0; i < 100; ++i) {
                std::lock_guard<std::mutex> lock(gen.mutex());
                // Advance the Philox counter and collect the new offset.
                local.push_back(gen.counter());
                gen.set_counter(gen.counter() + 1);
            }
            std::lock_guard<std::mutex> g(collect_mu);
            all_samples.insert(all_samples.end(), local.begin(), local.end());
        });
    }
    for (auto& th : threads) th.join();

    // No two locked increments should yield the same offset.
    std::sort(all_samples.begin(), all_samples.end());
    auto last = std::unique(all_samples.begin(), all_samples.end());
    EXPECT_EQ(last, all_samples.end())
        << "duplicate Philox offsets observed — Generator mutex did not "
           "serialize updates";
}
