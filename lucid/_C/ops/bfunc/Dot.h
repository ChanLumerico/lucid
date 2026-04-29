#pragma once

// =====================================================================
// dot product. 1-D × 1-D returns a scalar; N-D × M-D follows numpy.dot
// semantics (sum over last axis of a, second-to-last of b).
// Backward implemented for 1-D × 1-D and 2-D × 2-D.
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Dot.
LUCID_API TensorImplPtr dot_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
