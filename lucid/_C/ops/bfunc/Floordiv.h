#pragma once

// =====================================================================
// floordiv: floor(a / b), output dtype = Int (numpy/PyTorch convention).
// Forward only — has_gradient=False in the Python reference.
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Floordiv.
LUCID_API TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
