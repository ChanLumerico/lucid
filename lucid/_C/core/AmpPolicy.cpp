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

AutocastGuard::~AutocastGuard() {
    g_active = prev_active_;
    g_target = prev_dtype_;
}

}  // namespace amp
}  // namespace lucid
