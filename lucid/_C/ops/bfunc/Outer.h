#pragma once

// =====================================================================
// outer product (1-D × 1-D → 2-D).
// Backward: da = grad @ b, db = grad.T @ a.
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Outer.
LUCID_API TensorImplPtr outer_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
