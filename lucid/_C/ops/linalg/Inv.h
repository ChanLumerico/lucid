// lucid/_C/ops/linalg/Inv.h
//
// Matrix inverse op: given a square matrix A, compute A⁻¹.
//
// Forward dispatch goes to IBackend::linalg_inv(), which uses LAPACK's
// dgetrf (LU factorisation with partial pivoting) followed by dgetri
// (inversion from the LU factors) on the CPU path, and
// mlx::core::linalg::inv on the GPU path.
// The output shape is identical to the input shape.
//
// Mathematical background (Matrix Cookbook §9.1.3):
//   If A is invertible, A A⁻¹ = I.  Differentiating both sides:
//     dA A⁻¹ + A d(A⁻¹) = 0
//     d(A⁻¹) = -A⁻¹ dA A⁻¹
//   In gradient notation (upstream gradient G = ∂L/∂(A⁻¹)):
//     ∂L/∂A = -(A⁻¹)ᵀ G (A⁻¹)ᵀ
//
// Batched inputs: the last two dimensions are the matrix axes; all leading
// dimensions are treated as independent batch dimensions.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the matrix inverse backward pass.
//
// Invariant: saved_output_ holds the Storage of the forward result A⁻¹.
// This avoids recomputing the inverse during the backward pass, at the cost
// of keeping one extra matrix-shaped allocation alive in the graph.
//
// Backward formula (Matrix Cookbook §9.1.3):
//   Let B = A⁻¹ (the saved forward output) and G = ∂L/∂B (grad_out).
//   Then:
//     ∂L/∂A = -Bᵀ G Bᵀ
//
// The formula is implemented using existing matmul_op and neg_op so that
// second-order gradients (e.g. for Hessian computation) flow through the
// backward graph automatically at no extra implementation cost.
//
// FuncOp<InvBackward, 1> signals that this node has one differentiable input.
class LUCID_API InvBackward : public FuncOp<InvBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Compute and return the gradient for the single input A.
    //
    // Returns a one-element vector {∂L/∂A} aligned to the single input slot.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Compute A⁻¹ for the given square float tensor.
//
// Validates that a is at least 2-D, square, and float-typed, then dispatches
// to the backend.  Wires InvBackward into the autograd graph when grad mode
// is active.  saved_output_ on the backward node is set to the forward result
// so the backward pass can reuse B = A⁻¹ without a second inversion.
LUCID_API TensorImplPtr inv_op(const TensorImplPtr& a);

}  // namespace lucid
