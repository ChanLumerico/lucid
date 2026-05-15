// lucid/_C/ops/ufunc/Softmax.h
//
// Backward node and entry point for numerically stable softmax along a given
// axis.  SoftmaxBackward inherits from FuncOp (not the standard UnaryOp CRTP)
// because the forward and backward implementations are both non-trivial and
// hand-written rather than following the simple dispatch/grad_formula pattern.
//
// The backward uses the well-known Jacobian-vector product identity:
//   dL/dx = p * (dL/dy - sum_j(p_j * dL/dy_j))
//   where p = softmax(x) and the inner sum contracts along the softmax axis.
// This avoids forming the full Jacobian matrix.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Backward node for softmax along a single axis.
//
// Invariants:
//   - saved_output_ holds the softmax probabilities p = softmax(x, axis_),
//     saved during forward() so apply() does not need to re-run softmax.
//   - axis_ is normalised to a non-negative index by forward() before being
//     stored; apply() can use it directly.
//   - ForceFP32 is used to avoid probability underflow in float16.
class LUCID_API SoftmaxBackward : public FuncOp<SoftmaxBackward, 1> {
public:
    static const OpSchema schema_v1;
    int axis_ = -1;
    // Validates axis, dispatches softmax, saves output, and wires autograd.
    static TensorImplPtr forward(const TensorImplPtr& a, int axis);
    // Computes dL/dx = p*(dL/dy - dot(dL/dy, p)) along axis_ via the backend.
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr softmax_op(const TensorImplPtr& a, int axis);

// Backward node for numerically stable log-softmax along a single axis.
//
// Forward:  y = log_softmax(x, axis) = x - log(sum(exp(x), axis))  [stable]
// Backward: dL/dx = dL/dy - exp(y) * sum(dL/dy, axis)
//           where exp(y) = softmax(x), the probabilities.
class LUCID_API LogSoftmaxBackward : public FuncOp<LogSoftmaxBackward, 1> {
public:
    static const OpSchema schema_v1;
    int axis_ = -1;
    static TensorImplPtr forward(const TensorImplPtr& a, int axis);
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr log_softmax_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
