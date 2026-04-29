#pragma once

// =====================================================================
// Lucid C++ engine — Automatic Mixed Precision (AMP) policy.
// =====================================================================
//
// Each op declares one AmpPolicy in its OpSchema. When the user enters
// `lucid.amp.autocast(dtype=Float16)`, the CRTP forward path consults the
// active autocast dtype and the op's policy, then conditionally inserts a
// cast op into the autograd graph. The cast itself is a regular op (Phase
// 3.2 implements `cast.h`); AMP just decides when to insert it.
//
// Policy semantics (matches PyTorch torch.amp):
//   - Promote     : cast inputs to autocast dtype if higher-precision; used
//                   for matmul, conv, linear — fp16 is fine, sometimes faster.
//   - KeepInput   : do not auto-cast; used for pool, dropout, elementwise
//                   shape ops (lossless under any precision).
//   - ForceFP32   : always cast inputs to FP32 even under autocast; used
//                   for softmax, layernorm, loss reductions where fp16
//                   accumulators lose precision.
//
// Threading: AutocastGuard sets a thread-local target dtype. Crossing thread
// boundaries does not propagate autocast — start a new guard per worker.
//
// Layer: core/.

#include <optional>

#include "../api.h"
#include "Dtype.h"

namespace lucid {

enum class AmpPolicy : std::uint8_t {
    Promote,
    KeepInput,
    ForceFP32,
};

/// Amp policy name.
LUCID_API const char* amp_policy_name(AmpPolicy p);

namespace amp {

/// Returns the active autocast target dtype if `AutocastGuard` is in scope on
/// the current thread, else `std::nullopt`.
LUCID_API std::optional<Dtype> active_dtype();

/// Returns true iff an `AutocastGuard` is on the stack on this thread.
LUCID_API bool is_active();

/// RAII guard. Phase 3.0 only declares the type; ops actually consult it from
/// Phase 3.5 (NN ops) onward — until then the policy is recorded but no cast
/// is inserted (no Cast op exists yet).
class LUCID_API AutocastGuard {
public:
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
