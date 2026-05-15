// lucid/_C/ops/linalg/Det.h
//
// Scalar determinant op: given a square matrix A, compute det(A) as a scalar
// (or a batch of scalars for batched inputs).
//
// Forward dispatch goes to IBackend::linalg_det(), which uses LAPACK's
// dgetrf (LU factorisation with partial pivoting, then multiplies the diagonal
// of U and accounts for the sign of the permutation) on the CPU path, and
// mlx::core::linalg::det on the GPU path.
// The output shape equals the input shape with the last two dimensions removed.
//
// Mathematical background (Jacobi's formula / Matrix Cookbook §9.2.1):
//   d det(A) = det(A) tr(A⁻¹ dA)
//   In gradient notation (upstream scalar gradient g = ∂L/∂det(A)):
//     ∂L/∂A = det(A) · g · (A⁻¹)ᵀ
//
// The determinant collapses a matrix to a scalar, so the backward must
// broadcast (A⁻¹)ᵀ — which is matrix-shaped — against the scalar factor.
// This is handled by broadcast_to_op in the backward implementation.
//
// Note: det is numerically unstable for large matrices; log-determinant
// (logdet or slogdet) is preferred in practice but is not yet implemented.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the determinant backward pass.
//
// Invariants:
//   saved_inputs_[0] = Storage of the input matrix A (needed to call inv_op)
//   saved_output_    = Storage of det(A) (the scalar-per-batch output)
//
// Both tensors are saved because the backward formula requires both:
//   - A is needed to form (A⁻¹)ᵀ via inv_op + mT_op.
//   - det(A) is the multiplicative scale factor.
//
// Backward formula (Jacobi's formula):
//   Let d = det(A), g = ∂L/∂d, and G = g · d · (A⁻¹)ᵀ.
//   Then ∂L/∂A = broadcast(d * g, input_shape) * (A⁻¹)ᵀ.
//
// The scalar (d * g) is broadcast to the full matrix shape before the
// elementwise multiply so that batch dimensions are handled correctly.
//
// FuncOp<DetBackward, 1> signals that this node has one differentiable input.
class LUCID_API DetBackward : public FuncOp<DetBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Compute and return the gradient for the single input A.
    //
    // Returns a one-element vector {∂L/∂A} aligned to the single input slot.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Compute det(A) for the given square float tensor.
//
// Validates that a is at least 2-D, square, and float-typed, then dispatches
// to the backend.  Wires DetBackward into the autograd graph when grad mode
// is active; saves both the input A (for forming the inverse in backward) and
// the scalar output det(A) (the multiplicative factor in the gradient).
LUCID_API TensorImplPtr det_op(const TensorImplPtr& a);

}  // namespace lucid
