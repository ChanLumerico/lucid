// lucid/_C/ops/utils/MaskedSelect.h
// Boolean masked selection: returns a 1-D tensor of elements where mask==true.
#pragma once
#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {
LUCID_API TensorImplPtr masked_select_op(const TensorImplPtr& a, const TensorImplPtr& mask);
}  // namespace lucid
