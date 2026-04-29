#pragma once

// =====================================================================
// inner: last-axis contraction.
// out_shape = a.shape[:-1] + b.shape[:-1]; requires a.shape[-1] == b.shape[-1].
// Forward only — backward deferred (express as tensordot).
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Inner.
LUCID_API TensorImplPtr inner_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
