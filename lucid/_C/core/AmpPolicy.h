// lucid/_C/core/AmpPolicy.h
//
// Automatic Mixed Precision (AMP) policy and thread-local autocast state.
//
// AmpPolicy is stored per-op in OpSchema and describes how that op should
// treat its inputs when AMP is active:
//
//   Promote   — cast inputs to the active autocast dtype.  On CPU, F16 is
//               silently promoted to F32 because Accelerate does not support
//               native F16 arithmetic.
//   KeepInput — always execute at the input dtype regardless of the autocast
//               setting (e.g. loss scaling, batch norm stats).
//   ForceFP32 — always execute at F32 for numerical stability (e.g. softmax,
//               log, exponents).
//
// amp::AutocastGuard is the RAII equivalent of torch.autocast.  It saves and
// restores the per-thread active dtype on construction / destruction.
//
// Thread safety: all state is thread_local; no locking is required.

#pragma once

#include <optional>

#include "../api.h"
#include "Dtype.h"

namespace lucid {

// Controls how an op selects its computation dtype when AMP is active.
enum class AmpPolicy : std::uint8_t {
    Promote,    // Use autocast dtype (CPU F16 → F32).
    KeepInput,  // Ignore autocast; always use the input tensor's dtype.
    ForceFP32,  // Always use F32, regardless of autocast setting.
};

// Returns the human-readable name of the policy for diagnostics.
LUCID_API const char* amp_policy_name(AmpPolicy p);

namespace amp {

// Returns the active autocast dtype on the current thread, or std::nullopt
// when AMP is not active.
LUCID_API std::optional<Dtype> active_dtype();

// Returns true when an AutocastGuard is live on the current thread.
LUCID_API bool is_active();

// RAII guard that activates AMP with the specified target dtype for the
// lifetime of the object.  Saves and restores the previous active/dtype
// state, so guards can be nested.
class LUCID_API AutocastGuard {
public:
    // Activates AMP with target as the effective compute dtype.
    explicit AutocastGuard(Dtype target);
    ~AutocastGuard();

    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;

private:
    bool prev_active_;
    Dtype prev_dtype_;
};

}  // namespace amp
}  // namespace lucid
