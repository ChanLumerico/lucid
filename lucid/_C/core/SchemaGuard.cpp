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
        if (!schema.determinism_note.empty()) {
            msg += " (";
            msg += schema.determinism_note;
            msg += ")";
        }
        ErrorBuilder(schema.name).fail(msg);
    }
}

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

    if (!amp::is_active()) {
        effective_dtype_ = input_dtype;
        return;
    }

    const Dtype autocast_dt = *amp::active_dtype();
    switch (schema.amp_policy) {
    case AmpPolicy::Promote: {
        const bool cpu_f16 = (device == Device::CPU && autocast_dt == Dtype::F16);
        effective_dtype_ = cpu_f16 ? Dtype::F32 : autocast_dt;
        break;
    }
    case AmpPolicy::KeepInput:
        effective_dtype_ = input_dtype;
        break;
    case AmpPolicy::ForceFP32:
        effective_dtype_ = Dtype::F32;
        break;
    }
}

}  // namespace lucid
