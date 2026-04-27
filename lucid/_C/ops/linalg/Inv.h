#pragma once

// =====================================================================
// inv: matrix inverse. GPU only (routed through MLX CPU stream).
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr inv_op(const TensorImplPtr& a);

}  // namespace lucid
