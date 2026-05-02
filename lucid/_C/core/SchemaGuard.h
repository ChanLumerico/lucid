#pragma once

#include "../api.h"
#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Dtype.h"
#include "OpSchema.h"
#include "fwd.h"

namespace lucid {

LUCID_API void check_schema_determinism(const OpSchema& schema);

class LUCID_API SchemaGuard {
public:
    SchemaGuard(const OpSchema& schema, Dtype input_dtype, Device device = Device::CPU);

    Dtype effective_dtype() const noexcept { return effective_dtype_; }

private:
    Dtype effective_dtype_;
};

}  // namespace lucid
