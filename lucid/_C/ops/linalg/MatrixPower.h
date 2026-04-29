#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Matrix power.
LUCID_API TensorImplPtr matrix_power_op(const TensorImplPtr& a, int n);

}  // namespace lucid
