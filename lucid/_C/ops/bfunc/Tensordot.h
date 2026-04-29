#pragma once

// =====================================================================
// tensordot: Einstein-style multi-axis contraction.
// =====================================================================

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Tensordot.
LUCID_API TensorImplPtr tensordot_op(const TensorImplPtr& a,
                                     const TensorImplPtr& b,
                                     std::vector<int> axes_a,
                                     std::vector<int> axes_b);

}  // namespace lucid
