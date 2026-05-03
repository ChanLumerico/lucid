// lucid/_C/nn/Linear.h
//
// Autograd-aware fully-connected (linear) layer: y = x @ W^T + b.
// Exposes a single backward node (LinearBackward) and the free-function
// entry point linear_op() that the Python binding calls.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the linear operation.
//
// Inherits FuncOp<LinearBackward, 3> which wires three saved-input slots
// (x, W, b) and owns the shapes needed to reconstruct gradients.
// Thread safety: instances are created once during forward and consumed once
// during backward; no concurrent access is expected.
class LUCID_API LinearBackward : public FuncOp<LinearBackward, 3> {
public:
    // Registered schema: name="linear", version 1, AmpPolicy::Promote.
    static const OpSchema schema_v1;

    // Execute the forward pass and, when autograd is enabled, attach this
    // node as the grad_fn of the output tensor.
    // x      – input of shape (..., K).
    // W      – weight matrix of shape (N, K); transposed internally.
    // b      – bias of shape (N,).
    // Returns output of shape (..., N).
    // Throws ShapeMismatch if W.shape[1] != x.last_dim or b.shape[0] != N.
    static TensorImplPtr
    forward(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b);

    // Compute gradients for x, W, and b from grad_out.
    // dx = grad_out @ W, dW = grad_out^T @ x_flat, db = sum(grad_out, 0).
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point: delegates to LinearBackward::forward.
LUCID_API TensorImplPtr linear_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b);

}  // namespace lucid
