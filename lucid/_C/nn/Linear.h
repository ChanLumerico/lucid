#pragma once

// =====================================================================
// Lucid C++ engine — fused linear (linear + bias).
// =====================================================================
//
//   linear(x, W, b) = x @ W^T + b
//
// Where:
//   x : (batch..., in_features)        — supports an arbitrary number of
//                                         leading dims, flattened to 2D
//                                         internally for sgemm
//   W : (out_features, in_features)
//   b : (out_features,)
//
// Output: (batch..., out_features)
//
// Backward:
//   dx = grad @ W            : (batch..., in)
//   dW = grad^T @ x          : (out, in)        (sums over batch)
//   db = grad.sum(axis=batch dims)              : (out,)
//
// AMP policy: Promote — fp16 matmul is fine on AMX.
// Layer: autograd/ops/nn/.

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

/// Autograd backward node for Linear.
class LUCID_API LinearBackward : public FuncOp<LinearBackward, 3> {
public:
    static const OpSchema schema_v1;
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& W,
                                 const TensorImplPtr& b);
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Linear.
LUCID_API TensorImplPtr linear_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b);

}  // namespace lucid
