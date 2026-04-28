#pragma once

// =====================================================================
// var: variance reduction (sample variance, ddof=0).
// Backward: dx = (2/N) * (x - mean) * broadcast(grad).
// =====================================================================

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr var_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

}  // namespace lucid
