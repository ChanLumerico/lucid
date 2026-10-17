// lucid/_C/core/AmpPolicy.cpp
//
// The compiled policy table, thread-local AMP state and the AutocastGuard
// implementation.  The AMP state is thread_local so different Python
// threads or OpenMP worker threads can each have independent settings.

#include "AmpPolicy.h"

#include <algorithm>
#include <iterator>
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
    case AmpPolicy::PromoteWidest:
        return "PromoteWidest";
    }
    ErrorBuilder("amp_policy_name").fail("unknown AmpPolicy");
}

namespace amp {

namespace {

// The spellings ``AmpPolicyTable.def`` uses for its columns.
namespace column {
constexpr AmpPolicy LOWER = AmpPolicy::Promote;
constexpr AmpPolicy FP32 = AmpPolicy::ForceFP32;
constexpr AmpPolicy KEEP = AmpPolicy::KeepInput;
constexpr AmpPolicy WIDEST = AmpPolicy::PromoteWidest;
}  // namespace column

constexpr PolicyRow kRows[] = {
#define LUCID_AMP_ROW(name, metal, cpu) PolicyRow{#name, column::metal, column::cpu},
#include "AmpPolicyTable.def"
#undef LUCID_AMP_ROW
};

// ``policy_for`` binary-searches the rows, so an out-of-order or repeated
// name would make some op silently fall back to KeepInput.
constexpr bool rows_sorted_and_unique() {
    for (std::size_t i = 1; i < std::size(kRows); ++i) {
        if (!(kRows[i - 1].name < kRows[i].name))
            return false;
    }
    return true;
}
static_assert(rows_sorted_and_unique(),
              "AmpPolicyTable.def: rows must be sorted by name, each name once");

}  // namespace

AmpPolicy policy_for(std::string_view op, Device device) {
    const auto* const end = std::end(kRows);
    const auto* const it = std::lower_bound(
        std::begin(kRows), end, op,
        [](const PolicyRow& row, std::string_view name) { return row.name < name; });
    if (it == end || it->name != op)
        return AmpPolicy::KeepInput;
    return device == Device::CPU ? it->cpu : it->metal;
}

std::span<const PolicyRow> policy_table() {
    return kRows;
}

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
