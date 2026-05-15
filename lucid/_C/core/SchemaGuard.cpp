// lucid/_C/core/SchemaGuard.cpp
//
// Implementation of the determinism gate and AMP dtype-resolution logic.
// This translation unit is intentionally small: the two functions share the
// same determinism check, but check_schema_determinism() is provided as a
// standalone entry point for callers that need only the determinism gate
// without spinning up a full SchemaGuard (e.g. custom-function wrappers).

#include "SchemaGuard.h"

#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Error.h"
#include "ErrorBuilder.h"

namespace lucid {

void check_schema_determinism(const OpSchema& schema) {
    if (Determinism::is_enabled() && !schema.deterministic) {
        std::string msg = "non-deterministic op called under set_deterministic(True)";
        // Append the op-specific note (e.g. "uses atomic scatter-add") so the
        // user understands which aspect of the op causes non-determinism.
        if (!schema.determinism_note.empty()) {
            msg += " (";
            msg += schema.determinism_note;
            msg += ")";
        }
        ErrorBuilder(schema.name).fail(msg);
    }
}

// Performs the determinism check first (cheapest path — no memory reads beyond
// the atomic flag) and then resolves the effective dtype under AMP.
SchemaGuard::SchemaGuard(const OpSchema& schema, Dtype input_dtype, Device device) {
    if (Determinism::is_enabled() && !schema.deterministic) {
        std::string msg = "non-deterministic op called under set_deterministic(True)";
        if (!schema.determinism_note.empty()) {
            msg += " (";
            msg += schema.determinism_note;
            msg += ")";
        }
        ErrorBuilder(schema.name).fail(msg);
    }

    // Fast path: no AMP active — use the tensor's own dtype unchanged.
    if (!amp::is_active()) {
        effective_dtype_ = input_dtype;
        return;
    }

    const Dtype autocast_dt = *amp::active_dtype();
    switch (schema.amp_policy) {
    case AmpPolicy::Promote: {
        // Accelerate does not support native F16 arithmetic on the CPU stream;
        // demote to F32 instead of propagating an unsupported dtype downstream.
        const bool cpu_f16 = (device == Device::CPU && autocast_dt == Dtype::F16);
        effective_dtype_ = cpu_f16 ? Dtype::F32 : autocast_dt;
        break;
    }
    case AmpPolicy::KeepInput:
        // Ops like batch-norm running-statistics accumulators must stay at their
        // natural precision regardless of the outer autocast context.
        effective_dtype_ = input_dtype;
        break;
    case AmpPolicy::ForceFP32:
        // Numerically sensitive ops (softmax, log, exp) always run at F32 to
        // avoid catastrophic cancellation in reduced-precision formats.
        effective_dtype_ = Dtype::F32;
        break;
    }
}

}  // namespace lucid
