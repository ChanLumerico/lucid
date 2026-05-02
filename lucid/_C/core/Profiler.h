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

struct LUCID_API OpEvent {
    std::string name;
    Device device;
    Dtype dtype;
    Shape shape;
    std::int64_t time_ns = 0;
    std::int64_t memory_delta_bytes = 0;
    std::int64_t flops = 0;
};

class LUCID_API Profiler {
public:
    void start();
    void stop();
    bool is_active() const { return active_; }
    void clear();
    std::vector<OpEvent> events() const;

    void record(OpEvent event);

private:
    bool active_ = false;
    std::vector<OpEvent> events_;
    mutable std::mutex mu_;
};

LUCID_API Profiler* current_profiler();

LUCID_API void set_current_profiler(Profiler* p);

class LUCID_API OpScope {
public:
    OpScope(std::string_view name, Device device, Dtype dtype, Shape shape);
    ~OpScope();

    OpScope(const OpScope&) = delete;
    OpScope& operator=(const OpScope&) = delete;

    void set_flops(std::int64_t f);

private:
    Profiler* sink_;
    OpEvent event_;
    std::chrono::steady_clock::time_point start_time_;
    std::size_t start_memory_bytes_;
};

}  // namespace lucid
