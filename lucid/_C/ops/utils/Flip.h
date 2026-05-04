// lucid/_C/ops/utils/Flip.h
// Reverses a tensor along one or more axes.  Equivalent to numpy.flip / torch.flip.
#pragma once
#include <vector>
#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {
LUCID_API TensorImplPtr flip_op(const TensorImplPtr& a, std::vector<int> dims);
}  // namespace lucid
