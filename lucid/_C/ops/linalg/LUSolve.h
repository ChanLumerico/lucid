// lucid/_C/ops/linalg/LUSolve.h
#pragma once
#include "../../api.h"
#include "../../core/fwd.h"
namespace lucid {

LUCID_API TensorImplPtr lu_solve_op(const TensorImplPtr& LU,
                                    const TensorImplPtr& pivots,
                                    const TensorImplPtr& b);
}  // namespace lucid
