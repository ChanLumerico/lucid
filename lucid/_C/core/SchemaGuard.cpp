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
#include "Dtype.h"
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

    // Fast path: no AMP active — use the tensor's own dtype unchanged,
    // unless the op cannot express its answer in that dtype at all.
    if (!amp::is_active()) {
        // ``ForceFP32`` says this op needs real numbers: exp, log, sqrt,
        // tanh, sigmoid, erfinv.  It was only consulted inside an autocast
        // scope, so outside one an integer input took two different wrong
        // roads — half the family computed in the integer type and
        // truncated (``exp(1)`` answered ``2``), and the other half raised
        // ``NotImplementedError`` because no integer kernel existed.
        //
        // The reference framework promotes to float32 for all of them, and
        // that is the only answer that is not simply wrong: the value of
        // ``exp(1)`` is not an integer. Promoting here covers the whole
        // family at once rather than op by op, and leaves the *result*
        // dtype float, which is what the caller then observes.
        // ``is_integral`` deliberately excludes Bool, so it is named here:
        // the reference promotes a bool input for these ops too, and
        // ``sqrt(True)`` is 1.0 rather than a type error.
        if ((schema.real_valued || schema.amp_policy == AmpPolicy::ForceFP32)
            && (is_integral(input_dtype) || input_dtype == Dtype::Bool))
            effective_dtype_ = Dtype::F32;
        else
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
