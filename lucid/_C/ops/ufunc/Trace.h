#pragma once

// =====================================================================
// trace: sum of the main diagonal (last 2 axes).
// Backward (2-D inputs): dx = eye * grad.
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Trace.
LUCID_API TensorImplPtr trace_op(const TensorImplPtr& a);

}  // namespace lucid
