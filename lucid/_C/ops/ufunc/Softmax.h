#pragma once

// =====================================================================
// Lucid C++ engine — softmax (axis-aware, numerically stable).
// =====================================================================
//
//   softmax(x, axis) = exp(x - max(x, axis)) / sum(exp(...), axis)
//
// Forward subtracts max along axis to avoid overflow. Output shape == input.
// Saves output `z`; backward formula:
//
//   dx = z * (g - sum(g * z, axis, keepdim=True))
//
// AMP policy: ForceFP32 (softmax is precision-sensitive).
// Layer: autograd/ops/unary/.

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Autograd backward node for Softmax.
class LUCID_API SoftmaxBackward : public FuncOp<SoftmaxBackward, 1> {
public:
    static const OpSchema schema_v1;
    int axis_ = -1;
    static TensorImplPtr forward(const TensorImplPtr& a, int axis);
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Softmax.
LUCID_API TensorImplPtr softmax_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
