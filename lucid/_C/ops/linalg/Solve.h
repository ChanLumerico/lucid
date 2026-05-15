// lucid/_C/ops/linalg/Solve.h
//
// Linear system solve op: given square matrix A and right-hand side B,
// compute X = A⁻¹ B without explicitly forming the inverse.
//
// Compared to computing inv(A) and then multiplying, solve() is more
// numerically stable and roughly twice as fast (LU factorisation is shared).
//
// Forward dispatch goes to IBackend::linalg_solve(), which uses LAPACK's
// dgesv (LU factorisation with partial pivoting + triangular solves) on the
// CPU path, and mlx::core::linalg::solve on the GPU path.
// The output shape equals the shape of B.
//
// Mathematical background:
//   AX = B  =>  X = A⁻¹ B
//   Differentiating with respect to B (holding A fixed):
//     A dX = dB  =>  dX/dB = A⁻¹  =>  ∂L/∂B = A⁻ᵀ ∂L/∂X  (i.e. solve(Aᵀ, G))
//   Differentiating with respect to A (holding B fixed):
//     dA X + A dX = 0  =>  dX = -A⁻¹ dA X
//     ∂L/∂A = -(∂L/∂B) Xᵀ  (where ∂L/∂B = solve(Aᵀ, G) from above)
//
// Batched inputs: the last two dimensions of A and B are the matrix/vector
// axes; all leading dimensions are treated as independent batch dimensions.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the solve backward pass.
//
// Invariants:
//   saved_inputs_[0] = Storage of A   (needed to form solve(Aᵀ, G))
//   saved_output_    = Storage of X = A⁻¹ B  (needed to form dA = -dB Xᵀ)
//
// Both saved tensors are required:
//   - A is used in the solve(Aᵀ, G) call that computes ∂L/∂B.
//   - X is used to form the outer product -∂L/∂B @ Xᵀ for ∂L/∂A.
//
// Backward formulas (differentiate AX = B):
//   ∂L/∂B = solve(Aᵀ, G)   where G = ∂L/∂X is the upstream gradient
//   ∂L/∂A = -(∂L/∂B) Xᵀ
//
// The solve(Aᵀ, G) call reuses the existing solve_op, which means the LU
// factorisation of Aᵀ is performed fresh in the backward.  An optimisation
// would be to cache the LU factors from the forward pass, but this is not
// yet implemented.
//
// FuncOp<SolveBackward, 2> signals that this node has two differentiable
// inputs (A at index 0 and B at index 1), so apply() returns two Storages.
class LUCID_API SolveBackward : public FuncOp<SolveBackward, 2> {
public:
    static const OpSchema schema_v1;

    // Compute and return gradients {∂L/∂A, ∂L/∂B} for both inputs.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Solve the linear system AX = B for X.
//
// Validates that a is square, float-typed, and that a and b share dtype and
// device.  The output has the same shape as b.  Wires SolveBackward into the
// autograd graph when grad mode is active.
LUCID_API TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
