#pragma once

// =====================================================================
// Triangular masks: tril (keep below+diagonal+k), triu (above+diagonal+k).
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr tril_op(const TensorImplPtr& a, int k);
LUCID_API TensorImplPtr triu_op(const TensorImplPtr& a, int k);

}  // namespace lucid
