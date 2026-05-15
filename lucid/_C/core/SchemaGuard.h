// lucid/_C/core/SchemaGuard.h
//
// Runtime validation and dtype-resolution gate that sits at the entry point
// of every op dispatch.  SchemaGuard performs two jobs:
//
//   1. Determinism check — if Determinism::is_enabled() is true and the
//      schema's deterministic flag is false, it throws a LucidError before
//      any computation begins.
//
//   2. Effective dtype resolution — if AMP is active (amp::is_active()),
//      the input_dtype is adjusted according to schema.amp_policy:
//        Promote   — use the autocast dtype, but clamp CPU F16 → F32 because
//                    Accelerate does not support native F16 arithmetic.
//        KeepInput — ignore autocast; return input_dtype unchanged.
//        ForceFP32 — always return F32.
//
//      If AMP is not active, effective_dtype() == input_dtype.
//
// The free function check_schema_determinism() performs only the determinism
// check without constructing a full guard; useful in contexts where dtype
// resolution is handled separately.
//
// Typical op usage:
//   SchemaGuard guard(MyOp::schema_v1, input->dtype(), input->device());
//   const Dtype dt = guard.effective_dtype();
//   // ... allocate output and dispatch kernel at dt ...

#pragma once

#include "../api.h"
#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Dtype.h"
#include "OpSchema.h"
#include "fwd.h"

namespace lucid {

// Performs only the determinism check for schema; throws LucidError if
// Determinism::is_enabled() and schema.deterministic == false.
LUCID_API void check_schema_determinism(const OpSchema& schema);

// Combined determinism + AMP dtype-resolution guard.
//
// Constructing a SchemaGuard is cheap — it performs two conditional checks
// and a small computation.  It does not hold any locks or resources; it is
// not RAII in the "release on destruction" sense.
class LUCID_API SchemaGuard {
public:
    // Validates schema against the current runtime state and resolves the
    // effective computation dtype.  Throws LucidError if the op violates the
    // active determinism or AMP constraints.
    SchemaGuard(const OpSchema& schema, Dtype input_dtype, Device device = Device::CPU);

    // Returns the dtype that the op should use for its computation.
    // May differ from input_dtype when AMP is active and amp_policy is
    // Promote or ForceFP32.
    Dtype effective_dtype() const noexcept { return effective_dtype_; }

private:
    Dtype effective_dtype_;
};

}  // namespace lucid
