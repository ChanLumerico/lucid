#pragma once

// =====================================================================
// det: matrix determinant.
// Backward: dA = det(A) * ddet * A^{-T}
// =====================================================================

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Autograd backward node for Det.
class LUCID_API DetBackward : public FuncOp<DetBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Det.
LUCID_API TensorImplPtr det_op(const TensorImplPtr& a);

}  // namespace lucid
