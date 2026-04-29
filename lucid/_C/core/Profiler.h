#pragma once

// =====================================================================
// Lucid C++ engine — profiler (P8: timing + memory + flops per op).
// =====================================================================
//
// The CRTP `forward()` always opens an `OpScope`. When no Profiler is active
// on the current thread, OpScope is near-zero overhead (one thread-local
// pointer load + null check). When active, it records:
//   - elapsed time (ns, std::chrono::steady_clock)
//   - memory delta (current_bytes pre/post via MemoryTracker)
//   - flops (set by the op via OpScope::set_flops)
//
// Python API (Phase 5.10) wraps this:
//
//   with lucid.profiler() as prof:
//       loss.backward()
//   prof.report(top_k=20)
//
// Threading: each Profiler has an internal mutex; OpScope acquires it only
// when recording. Profilers are not shared across threads by default — each
// thread has its own active-profiler pointer.
//
// Layer: core/.

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "Shape.h"

namespace lucid {

/// OpEvent.
struct LUCID_API OpEvent {
    std::string name;
    Device device;
    Dtype dtype;
    Shape shape;
    std::int64_t time_ns = 0;
    std::int64_t memory_delta_bytes = 0;
    std::int64_t flops = 0;
};

/// Profiler.
class LUCID_API Profiler {
public:
    void start();
    void stop();
    bool is_active() const { return active_; }
    void clear();
    std::vector<OpEvent> events() const;

    // Called by ~OpScope when this profiler is the active one for the thread.
    void record(OpEvent event);

private:
    bool active_ = false;
    std::vector<OpEvent> events_;
    mutable std::mutex mu_;
};

/// Current profiler.
LUCID_API Profiler* current_profiler();
/// Set current profiler.
LUCID_API void set_current_profiler(Profiler* p);

/// OpScope.
class LUCID_API OpScope {
public:
    OpScope(std::string_view name, Device device, Dtype dtype, Shape shape);
    ~OpScope();

    OpScope(const OpScope&) = delete;
    OpScope& operator=(const OpScope&) = delete;

    void set_flops(std::int64_t f);

private:
    Profiler* sink_;  // null = no recording
    OpEvent event_;
    std::chrono::steady_clock::time_point start_time_;
    std::size_t start_memory_bytes_;
};

}  // namespace lucid
