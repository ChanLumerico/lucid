// lucid/_C/core/AmpPolicy.cpp
//
// Thread-local AMP state and AutocastGuard implementation.
// g_active tracks whether autocast is in effect; g_target holds the requested
// compute dtype (e.g. F16 for GPU half-precision training).  Both are
// thread_local so different Python threads or OpenMP worker threads can each
// have independent AMP settings.

#include "AmpPolicy.h"

#include <stdexcept>

#include "Error.h"
#include "ErrorBuilder.h"

namespace lucid {

const char* amp_policy_name(AmpPolicy p) {
    switch (p) {
    case AmpPolicy::Promote:
        return "Promote";
    case AmpPolicy::KeepInput:
        return "KeepInput";
    case AmpPolicy::ForceFP32:
        return "ForceFP32";
    }
    ErrorBuilder("amp_policy_name").fail("unknown AmpPolicy");
}

namespace amp {

namespace {
struct DeviceAmpState {
    bool active = false;
    Dtype target = Dtype::F32;
};

// Per-thread, per-device autocast flags.  Each device starts inactive so AMP is opt-in.
thread_local DeviceAmpState g_cpu_state{};
thread_local DeviceAmpState g_gpu_state{};

inline DeviceAmpState& state_for(Device device) noexcept {
    return (device == Device::CPU) ? g_cpu_state : g_gpu_state;
}
}  // namespace

std::optional<Dtype> active_dtype(Device device) {
    const auto& s = state_for(device);
    if (!s.active)
        return std::nullopt;
    return s.target;
}

std::optional<Dtype> active_dtype() {
    if (g_gpu_state.active)
        return g_gpu_state.target;
    if (g_cpu_state.active)
        return g_cpu_state.target;
    return std::nullopt;
}

bool is_active(Device device) {
    return state_for(device).active;
}

bool is_active() {
    return g_gpu_state.active || g_cpu_state.active;
}

AutocastGuard::AutocastGuard(Dtype target)
    : device_(std::nullopt),
      prev_cpu_active_(g_cpu_state.active),
      prev_cpu_dtype_(g_cpu_state.target),
      prev_gpu_active_(g_gpu_state.active),
      prev_gpu_dtype_(g_gpu_state.target) {
    if (!g_cpu_state.active && !g_gpu_state.active) {
        g_cpu_state.active = true;
        g_cpu_state.target = target;
        g_gpu_state.active = true;
        g_gpu_state.target = target;
    } else {
        if (g_cpu_state.active)
            g_cpu_state.target = target;
        if (g_gpu_state.active)
            g_gpu_state.target = target;
    }
}

AutocastGuard::AutocastGuard(Dtype target, Device device, bool enabled)
    : device_(device),
      prev_cpu_active_(g_cpu_state.active),
      prev_cpu_dtype_(g_cpu_state.target),
      prev_gpu_active_(g_gpu_state.active),
      prev_gpu_dtype_(g_gpu_state.target) {
    auto& s = state_for(device);
    s.active = enabled;
    if (enabled)
        s.target = target;
}

// Unconditionally restores the previous state so nested guards work correctly
// (e.g. an inner ForceFP32 guard nested inside an outer F16 guard).
AutocastGuard::~AutocastGuard() {
    if (device_.has_value()) {
        auto& s = state_for(*device_);
        if (*device_ == Device::CPU) {
            s.active = prev_cpu_active_;
            s.target = prev_cpu_dtype_;
        } else {
            s.active = prev_gpu_active_;
            s.target = prev_gpu_dtype_;
        }
    } else {
        g_cpu_state.active = prev_cpu_active_;
        g_cpu_state.target = prev_cpu_dtype_;
        g_gpu_state.active = prev_gpu_active_;
        g_gpu_state.target = prev_gpu_dtype_;
    }
}

}  // namespace amp
}  // namespace lucid
