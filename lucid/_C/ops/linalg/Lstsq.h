// lucid/_C/ops/linalg/Lstsq.h
#pragma once
#include <vector>
#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {
// Least-squares: min||AX-B||_2. Returns {solution}.
LUCID_API std::vector<TensorImplPtr> lstsq_op(const TensorImplPtr& a, const TensorImplPtr& b);
}  // namespace lucid
