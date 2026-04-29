#pragma once

// =====================================================================
// Prefix sums and products along a single axis.
//   cumsum(x, axis)  : prefix sum
//   cumprod(x, axis) : prefix product
// cumsum has a closed-form backward (reverse-cumsum-reverse); cumprod's
// backward is deferred (requires zero-safe division).
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Cumsum.
LUCID_API TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis);
/// Cumprod.
LUCID_API TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
