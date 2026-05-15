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
// Per-thread autocast flags.  g_active starts false so AMP is opt-in.
thread_local bool g_active = false;
thread_local Dtype g_target = Dtype::F32;
}  // namespace

std::optional<Dtype> active_dtype() {
    if (!g_active)
        return std::nullopt;
    return g_target;
}

bool is_active() {
    return g_active;
}

AutocastGuard::AutocastGuard(Dtype target) : prev_active_(g_active), prev_dtype_(g_target) {
    g_active = true;
    g_target = target;
}

// Unconditionally restores the previous state so nested guards work correctly
// (e.g. an inner ForceFP32 guard nested inside an outer F16 guard).
AutocastGuard::~AutocastGuard() {
    g_active = prev_active_;
    g_target = prev_dtype_;
}

}  // namespace amp
}  // namespace lucid
