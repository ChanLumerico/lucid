// lucid/_C/core/Profiler.h
//
// Lightweight op-level profiler and RAII scope recorder.
//
// Profiler is an event sink that collects OpEvent records while active.  It
// is not a global singleton; instead, a per-thread pointer (current_profiler())
// designates the active Profiler for the calling thread.  This design allows
// Python test code and benchmark harnesses to swap profilers without affecting
// other threads.
//
// OpScope is the RAII mechanism: constructing an OpScope captures a wall-clock
// start time and a memory baseline; the destructor computes the elapsed time
// and memory delta, then calls Profiler::record().  If no Profiler is active
// (current_profiler() == nullptr or Profiler::is_active() is false), OpScope
// becomes a near-zero-cost no-op.
//
// Thread safety:
//   Profiler::record(), start(), stop(), clear(), and events() are protected
//   by an internal mutex, making it safe to collect events from multiple
//   threads into a single Profiler.  The current_profiler() pointer itself is
//   thread-local, so different threads can each have their own active Profiler.

#pragma once

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

// Record of a single op execution captured by OpScope.
//
// time_ns is the wall-clock duration from OpScope construction to destruction.
// memory_delta_bytes is the change in MemoryTracker::current_bytes for
// event_.device between the same two points; it can be negative if the op
// freed memory (e.g. in-place ops that release a temporary).
// flops is set explicitly via OpScope::set_flops() and defaults to 0 if the
// op does not report a FLOPs count.
struct LUCID_API OpEvent {
    std::string name;
    Device device;
    Dtype dtype;
    Shape shape;
    std::int64_t time_ns = 0;
    std::int64_t memory_delta_bytes = 0;
    std::int64_t flops = 0;
};

// Thread-safe event collector for profiling op execution.
//
// Typical usage:
//   Profiler prof;
//   set_current_profiler(&prof);
//   prof.start();
//   // ... execute ops ...
//   prof.stop();
//   auto events = prof.events();
//   set_current_profiler(nullptr);
class LUCID_API Profiler {
public:
    // Enables event collection.  Subsequent OpScope destructions will call
    // record() and their events will be stored.
    void start();

    // Disables event collection.  Events already stored are not cleared.
    void stop();

    // Returns true while the profiler is recording.
    bool is_active() const { return active_; }

    // Removes all stored events.  Does not change the active/stopped state.
    void clear();

    // Returns a copy of the stored event list under the internal mutex.
    std::vector<OpEvent> events() const;

    // Appends event to the stored list if is_active().  Thread-safe.
    void record(OpEvent event);

private:
    bool active_ = false;
    std::vector<OpEvent> events_;
    mutable std::mutex mu_;
};

// Returns the Profiler currently registered for this thread, or nullptr.
LUCID_API Profiler* current_profiler();

// Registers p as the active Profiler for the calling thread.  Pass nullptr to
// deactivate profiling without destroying the Profiler object.
LUCID_API void set_current_profiler(Profiler* p);

// RAII scope that records one OpEvent into the current thread's Profiler.
//
// If current_profiler() is nullptr or the profiler is not active at
// construction time, this object does nothing and imposes no overhead on
// the destructor path (sink_ is set to nullptr on construction).
//
// Ownership: OpScope does not own the Profiler; set_current_profiler() must
// ensure the Profiler outlives any OpScope that might record into it.
class LUCID_API OpScope {
public:
    // Captures the start wall-clock time and memory baseline.  name, device,
    // dtype, and shape are stored in the pending OpEvent and later forwarded
    // to Profiler::record().
    OpScope(std::string_view name, Device device, Dtype dtype, Shape shape);

    // Finalises the event with elapsed time and memory delta, then records it.
    ~OpScope();

    OpScope(const OpScope&) = delete;
    OpScope& operator=(const OpScope&) = delete;

    // Records the estimated floating-point operation count for this scope.
    // Must be called before the scope exits; defaults to 0 if never called.
    void set_flops(std::int64_t f);

private:
    // Non-owning pointer to the active Profiler; nullptr if profiling is off.
    Profiler* sink_;
    OpEvent event_;
    std::chrono::steady_clock::time_point start_time_;
    // MemoryTracker::current_bytes at construction time, used to compute the
    // memory delta on destruction.
    std::size_t start_memory_bytes_;
};

}  // namespace lucid
