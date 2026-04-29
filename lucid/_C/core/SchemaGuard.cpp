#include "SchemaGuard.h"

#include "../backend/Dispatcher.h"
#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Error.h"
#include "ErrorBuilder.h"
#include "TensorImpl.h"

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
    // ---- 1. Determinism gate ----
    if (Determinism::is_enabled() && !schema.deterministic) {
        std::string msg = "non-deterministic op called under set_deterministic(True)";
        if (!schema.determinism_note.empty()) {
            msg += " (";
            msg += schema.determinism_note;
            msg += ")";
        }
        ErrorBuilder(schema.name).fail(msg);
    }

    // ---- 2. AMP dtype resolution ----
    if (!amp::is_active()) {
        effective_dtype_ = input_dtype;
        return;
    }

    const Dtype autocast_dt = *amp::active_dtype();
    switch (schema.amp_policy) {
        case AmpPolicy::Promote: {
            // CPU Accelerate has no F16 ops — fall back to F32 when autocast
            // requests F16 on CPU. GPU (MLX) handles F16 natively.
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

TensorImplPtr SchemaGuard::maybe_cast(const TensorImplPtr& t) const {
    if (!t || t->dtype() == effective_dtype_)
        return t;

    auto& be = backend::Dispatcher::for_device(t->device());
    Storage cast_storage = be.cast(t->storage(), t->shape(), t->dtype(), effective_dtype_);
    return std::make_shared<TensorImpl>(
        std::move(cast_storage), t->shape(), effective_dtype_, t->device(),
        /*requires_grad=*/false);
}

}  // namespace lucid
