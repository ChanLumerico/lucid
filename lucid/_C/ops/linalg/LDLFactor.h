// lucid/_C/ops/linalg/LDLFactor.h
#pragma once
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {
LUCID_API std::vector<TensorImplPtr> ldl_factor_op(const TensorImplPtr& a);
}  // namespace lucid
