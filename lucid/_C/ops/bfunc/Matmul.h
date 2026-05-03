// lucid/_C/ops/bfunc/Matmul.h
//
// Declares MatmulBackward, the autograd node for batched matrix multiplication,
// and the public free function matmul_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for batched matrix multiplication: C = A @ B.
//
// Both inputs must be at least 2-D.  Leading batch dimensions are broadcast
// using NumPy rules (plan_nd_matmul in kernel/primitives/BatchedMatmul.h
// computes the broadcast shapes and the M/K/N extents).
//
// Forward:  C[..., i, j] = Σ_k A[..., i, k] * B[..., k, j]
// Backward: dA = grad_out @ B^T   (shape: bcast_a_shape → original a.shape)
//           dB = A^T @ grad_out   (shape: bcast_b_shape → original b.shape)
//
// MatmulBackward is a FuncOp<..., 2> (alias for AutogradNode<..., 2>) rather
// than a BinaryOp because matmul requires a custom forward() that cannot fit
// into the generic BinaryKernel dispatch path.  forward() calls plan_nd_matmul,
// handles batch broadcasting, and wires the autograd graph manually.
class LUCID_API MatmulBackward : public FuncOp<MatmulBackward, 2> {
public:
    // Op registration metadata: name "matmul", schema version 1, dtype
    // promotion, deterministic.
    static const OpSchema schema_v1;

    // Validate inputs, plan the batched matrix multiply, execute the forward
    // kernel, and wire the autograd graph if gradient tracking is active.
    static TensorImplPtr forward(const TensorImplPtr& a, const TensorImplPtr& b);

    // Compute dA and dB from the upstream gradient.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Public entry point: compute A @ B with full batch broadcasting and autograd.
LUCID_API TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
