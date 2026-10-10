// lucid/_C/ops/linalg/Det.h
//
// Scalar determinant $\det(A)$ for square float matrices, with
// autograd via Jacobi's formula.
//
// Forward reduces the trailing two matrix axes of an
// ``(..., N, N)`` input to a single scalar per batch.  CPU dispatch
// is LAPACK ``*getrf`` (LU factorisation with partial pivoting),
// after which the determinant is the product of $U_{ii}$ adjusted by
// $\mathrm{sign}(\det P)$.  GPU dispatch is ``mlx::core::linalg::det``
// on the CPU stream (linalg carve-out, DEVELOPMENT.md §H3).
//
// Backward uses Jacobi's formula
// $$
//   d\,\det(A) = \det(A)\,\mathrm{tr}(A^{-1}\,dA)
//     \;\;\Longleftrightarrow\;\;
//     \frac{\partial L}{\partial A}
//       = \det(A)\,\frac{\partial L}{\partial \det}\,(A^{-1})^\top.
// $$
// $\det(A)\,A^{-\top}$ is the cofactor matrix $\mathrm{cof}(A)$, which
// exists at every $A$ although $A^{-1}$ does not.  When any matrix of the
// batch is singular ($|\det| < 100\,\varepsilon$) the backward forms
// $\mathrm{cof}(A)$ from the SVD instead ([[CofactorBackward]]), so the
// gradient there is the adjugate's transpose rather than an inverse
// failure.
//
// Note
// ----
// ``det`` over/underflows for moderate-to-large matrices; prefer
// ``slogdet`` there.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the determinant $\det(A)$.
//
// Implements Jacobi's formula by reusing differentiable primitives
// (``inv_op``, ``mT_op``, ``mul_op``, ``broadcast_to_op``) so the
// backward sub-graph is itself differentiable — second-order
// gradients (Hessians) work without additional code.
//
// Both inputs and outputs are saved: $A$ is required to form
// $(A^{-1})^\top$ in the backward and $\det(A)$ is the scalar
// multiplicative factor.
//
// Math
// ----
// $$
//   d = \det(A), \qquad g = \frac{\partial L}{\partial d},
// $$
// $$
//   \frac{\partial L}{\partial A}
//     = \mathrm{broadcast}(d \cdot g,\; \text{shape}(A))
//       \;\odot\; (A^{-1})^\top.
// $$
//
// Shape
// -----
// - ``A``: ``(..., N, N)``.
// - ``grad_out``: ``(...,)`` (matches forward output shape).
// - Output gradient: ``(..., N, N)``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"det"``, one input, ``AmpPolicy::KeepInput`` to prevent AMP
//     downcasting before the inverse call.
// saved_inputs_[0] : Storage
//     Storage of the input matrix $A$.
// saved_output_ : Storage
//     Storage of the scalar (per-batch) determinant $\det(A)$.
//
// References
// ----------
// Petersen & Pedersen, *The Matrix Cookbook* (2012), §9.2.1.
// Magnus & Neudecker, *Matrix Differential Calculus* (3rd ed.), §8.4.
//
// See Also
// --------
// InvBackward : Used internally to form $(A^{-1})^\top$.
class LUCID_API DetBackward : public FuncOp<DetBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Compute the gradient ``{∂L/∂A}`` from the upstream scalar gradient.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient $\partial L/\partial \det$ with shape
    //     matching ``out_shape_``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-entry vector ``{∂L/∂A}`` aligned with the one
    //     differentiable input slot.
    std::vector<Storage> apply(Storage grad_out) override;

    // The same gradient, recorded, with its factors recomputed from A.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Autograd node for the cofactor matrix $\mathrm{cof}(A) = \nabla \det(A)$
// in its singular-safe SVD form — what [[DetBackward]] records when the
// batch holds a singular matrix, so ``det`` differentiates twice there too.
//
// The cofactor map is the gradient of $\det$, so its vector-Jacobian
// product is the Hessian of $\det$ applied to $G$, and that Hessian is
// symmetric: the backward is the directional derivative of $\mathrm{cof}$
// along $G$.  With $A = U\,\mathrm{diag}(s)\,V^\top$, $\alpha =
// \det U \det V$ and $G' = U^\top G V$,
// $$
//   d\,\mathrm{cof}(A)[G] = \alpha\, U \left(
//     \mathrm{diag}(Q\,\mathrm{diag}(G')) - Q \odot G'^\top \right) V^\top,
//   \qquad Q_{ij} = [i \ne j] \prod_{m \ne i, j} s_m,
// $$
// which takes no division, so it holds at any rank.  A third derivative
// through it is not implemented.
class LUCID_API CofactorBackward : public FuncOp<CofactorBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<Storage> apply(Storage grad_out) override;
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Compute $\det(A)$ for a square float tensor.
//
// Validates that ``a`` is at least 2-D, square in the trailing two
// dimensions, and float-typed; dispatches to the backend; and wires
// ``DetBackward`` into the autograd graph (saving both $A$ and the
// scalar output) when grad mode is active.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input matrix of shape ``(..., N, N)``.
//
// Returns
// -------
// TensorImplPtr
//     Scalar-per-batch determinant of shape ``(...,)`` (the trailing
//     two matrix dimensions are removed).
//
// Shape
// -----
// - ``a``: ``(..., N, N)``.
// - Output: ``(...,)`` (rank decreases by 2; a single 2-D input
//   yields a 0-D scalar).
//
// Raises
// ------
// ValueError
//     If ``a`` is fewer than 2 dimensions, non-square, or non-float.
// LinAlgError
//     (At backend level.) If $A$ is singular — LU returns a zero
//     pivot.
//
// Notes
// -----
// For ill-conditioned $A$ prefer ``slogdet`` (sign + log) when it
// becomes available; the magnitude of ``det`` can over/underflow long
// before $A$ is numerically singular.
//
// See Also
// --------
// inv_op : Matrix inverse, used in the backward.
LUCID_API TensorImplPtr det_op(const TensorImplPtr& a);

}  // namespace lucid
