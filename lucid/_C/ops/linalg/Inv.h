#pragma once

// =====================================================================
// inv: matrix inverse. CPU via LAPACK, GPU via MLX.
// Backward: dA = -B^T @ dB @ B^T  where B = inv(A).
// =====================================================================

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Autograd backward node for Inv.
class LUCID_API InvBackward : public FuncOp<InvBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Inv.
LUCID_API TensorImplPtr inv_op(const TensorImplPtr& a);

}  // namespace lucid
