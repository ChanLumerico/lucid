#pragma once

// =====================================================================
// solve: linear system Ax = B.
// Backward: dB = solve(A^T, dX),  dA = -dB @ X^T
// =====================================================================

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Autograd backward node for Solve.
class LUCID_API SolveBackward : public FuncOp<SolveBackward, 2> {
public:
    static const OpSchema schema_v1;
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Solve.
LUCID_API TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
