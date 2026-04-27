#include "Profiler.h"

#include "MemoryStats.h"

namespace lucid {

namespace {
thread_local Profiler* g_current = nullptr;
}  // namespace

Profiler* current_profiler() { return g_current; }
void set_current_profiler(Profiler* p) { g_current = p; }

void Profiler::start() {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = true;
}

void Profiler::stop() {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = false;
}

void Profiler::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    events_.clear();
}

std::vector<OpEvent> Profiler::events() const {
    std::lock_guard<std::mutex> lock(mu_);
    return events_;
}

void Profiler::record(OpEvent event) {
    std::lock_guard<std::mutex> lock(mu_);
    if (active_) events_.push_back(std::move(event));
}

OpScope::OpScope(std::string_view name, Device device, Dtype dtype, Shape shape)
    : sink_(g_current),
      event_{std::string(name), device, dtype, std::move(shape), 0, 0, 0},
      start_time_(std::chrono::steady_clock::now()),
      start_memory_bytes_(0) {
    if (sink_ && sink_->is_active()) {
        start_memory_bytes_ = MemoryTracker::get_stats(device).current_bytes;
    } else {
        sink_ = nullptr;  // disable bookkeeping path
    }
}

OpScope::~OpScope() {
    if (!sink_) return;
    const auto now = std::chrono::steady_clock::now();
    event_.time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         now - start_time_)
                         .count();
    const auto cur = MemoryTracker::get_stats(event_.device).current_bytes;
    event_.memory_delta_bytes =
        static_cast<std::int64_t>(cur) - static_cast<std::int64_t>(start_memory_bytes_);
    sink_->record(std::move(event_));
}

void OpScope::set_flops(std::int64_t f) { event_.flops = f; }

}  // namespace lucid
