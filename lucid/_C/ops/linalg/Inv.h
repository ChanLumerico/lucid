// lucid/_C/ops/linalg/Inv.h
//
// Matrix inverse $A^{-1}$ for batched square float matrices with
// closed-form autograd (Giles 2008).
//
// Forward factorises $A$ via LAPACK ``*getrf`` (LU with partial
// pivoting) and inverts via ``*getri`` on the CPU stream; the GPU
// stream dispatches to ``mlx::core::linalg::inv`` (MLX-on-CPU linalg
// carve-out, DEVELOPMENT.md §H3).  Batched inputs reduce over the
// last two dimensions; all leading dimensions are treated as
// independent batches.
//
// Backward applies the standard closed-form derivative
// $$
//   d(A^{-1}) = -A^{-1}\,dA\,A^{-1}
//     \;\;\Longrightarrow\;\;
//     \frac{\partial L}{\partial A}
//       = -(A^{-1})^\top\,\frac{\partial L}{\partial A^{-1}}\,(A^{-1})^\top
// $$
// reusing the saved $B = A^{-1}$ so no second factorisation is
// required at backward time.  The backward is composed from
// ``matmul_op`` and ``neg_op``, both of which are themselves
// differentiable — second-order gradients (Hessian-vector products)
// work without additional code.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the matrix inverse $A^{-1}$.
//
// The forward result $B = A^{-1}$ is captured in ``saved_output_``;
// no input is saved because the closed-form gradient only requires
// $B$.  This trades a single matrix-shaped retain (peak memory) for
// avoiding a second LU factorisation during backward.
//
// Math
// ----
// $$
//   B = A^{-1}, \qquad G = \frac{\partial L}{\partial B},
// $$
// $$
//   \frac{\partial L}{\partial A} = -B^\top\,G\,B^\top.
// $$
//
// Shape
// -----
// - ``A``: ``(..., N, N)``.
// - ``B = A^{-1}``: same shape as ``A``.
// - ``grad_out``: same shape as ``A``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"inv"``, one input, ``AmpPolicy::KeepInput`` so mixed-precision
//     training does not silently downcast the saved inverse.
// saved_output_ : Storage
//     Storage of the forward result $B = A^{-1}$ retained for the
//     backward.
//
// Notes
// -----
// - For very large matrices in deep graphs, holding $B$ alive until
//   backward increases peak memory; users wishing to trade memory
//   for compute would need to recompute the inverse (not currently
//   supported here).
// - Implementation uses ``matmul_op`` + ``neg_op`` so second-order
//   gradients flow through automatically.
//
// References
// ----------
// Giles, "Collected Matrix Derivative Results for Forward and
// Reverse Mode Algorithmic Differentiation" (Oxford NA-08/01, 2008),
// §2.2.3.
// Petersen & Pedersen, *The Matrix Cookbook* (2012), §9.1.3.
class LUCID_API InvBackward : public FuncOp<InvBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Compute the gradient ``{∂L/∂A}`` from the upstream gradient.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient $\partial L/\partial B$ matching the
    //     forward output shape.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-entry vector ``{∂L/∂A}`` aligned with the one
    //     differentiable input slot.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Compute $A^{-1}$ for a square float tensor.
//
// Validates that ``a`` is at least 2-D, square in the trailing two
// axes, and float-typed; dispatches to the backend; and wires
// ``InvBackward`` into the autograd graph (saving only the output)
// when grad mode is active.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input matrix of shape ``(..., N, N)``.
//
// Returns
// -------
// TensorImplPtr
//     Inverse $A^{-1}$ of identical shape to ``a``.
//
// Shape
// -----
// - ``a``: ``(..., N, N)``.
// - Output: same shape as ``a``.
//
// Raises
// ------
// ValueError
//     If ``a`` is not square, fewer than 2-D, or non-float.
// LinAlgError
//     (At backend level.) If $A$ is singular (LU produces a zero
//     pivot).
//
// See Also
// --------
// solve_op : Often a better alternative — solves $AX = B$ without
//     forming $A^{-1}$ explicitly, with better conditioning.
// det_op : Determinant, which uses ``inv_op`` in its backward.
LUCID_API TensorImplPtr inv_op(const TensorImplPtr& a);

}  // namespace lucid
