// lucid/_C/ops/linalg/LUSolve.h
//
// Triangular back-substitution using a pre-computed LU factorisation —
// solves $Ax = b$ given the packed $LU$ and pivot vector produced by
// [[lu_factor_op]].
//
// Splitting the factorisation step out from the solve step lets callers
// reuse a single LU decomposition across many right-hand sides without
// re-running the $O(n^3)$ factorisation, which is the dominant cost.
//
// Math
// ----
// With $PA = LU$ from [[lu_factor_op]], solving $Ax = b$ reduces to two
// triangular sweeps:
// $$
//   Ly = Pb, \qquad Ux = y
// $$
// Forward substitution applies $L^{-1}$ (and the permutation $P$);
// backward substitution applies $U^{-1}$.  Each sweep is $O(n^2)$.
//
// Notes
// -----
// - The CPU stream dispatches to LAPACK ``sgetrs_``/``dgetrs_`` via
//   ``IBackend::linalg_lu_solve()``.
// - ``b`` follows the solve family's right-hand-side contract
//   (``solve_rhs_contract``): ``(*, n, k)``, or a vector — ``(n,)`` or
//   exactly ``LU.shape[:-1]`` — with the batch axes of ``LU`` (and its
//   pivots) and ``b`` broadcast.  Any other shape raises ``ShapeMismatch``
//   before LAPACK is called.
// - Differentiable in ``LU`` and ``b`` through [[LUSolveBackward]].
//
// References
// ----------
// Anderson et al., *LAPACK Users' Guide* (3rd ed., SIAM, 1999),
// §2.5.1 "Solving Linear Systems".

#pragma once
#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
namespace lucid {

// Autograd node for $X = A^{-1} B$ with $A = P L U$ given packed as ``LU``,
// wired on the aligned ``(batch, n, n)`` factor and ``(batch, n, k)``
// right-hand side.  The pivots are an integer input with no gradient and
// are held in ``pivots_``.
//
// With $G = \partial L / \partial X$ and $Y = U^{-\top} G$:
// $$
//   \frac{\partial L}{\partial B} = A^{-\top} G = P\, L^{-\top} Y, \qquad
//   R = -Y X^\top,
// $$
// $$
//   \frac{\partial L}{\partial \mathrm{LU}} =
//     \mathrm{triu}(R) + \mathrm{tril}\!\left(L^{-\top} R\, U^\top, -1\right),
// $$
// the upper part being $\partial L / \partial U$ and the strict lower part
// $\partial L / \partial L$ (the unit diagonal of $L$ is not stored).  Every
// step is a recorded triangular solve or product, so the result
// differentiates again.
class LUCID_API LUSolveBackward : public FuncOp<LUSolveBackward, 2> {
public:
    static const OpSchema schema_v1;

    // The aligned ``(batch, n)`` pivot vector of the factor.
    TensorImplPtr pivots_;

    std::vector<Storage> apply(Storage grad_out) override;
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Solve $Ax = b$ given the packed LU factors and pivot vector.
//
// Parameters
// ----------
// LU : const TensorImplPtr&
//     Packed LU matrix of shape ``(..., n, n)`` as returned by
//     [[lu_factor_op]] — upper triangle holds $U$, strict lower
//     triangle holds the off-diagonal entries of $L$.
// pivots : const TensorImplPtr&
//     1-based pivot indices of shape ``LU.shape[:-1]`` and dtype ``I32``.
// b : const TensorImplPtr&
//     Right-hand side — see Shape.
//
// Returns
// -------
// TensorImplPtr
//     Solution tensor with the dtype of ``b``.
//
// Shape
// -----
// - ``LU``: ``(*, n, n)``.
// - ``pivots``: ``LU.shape[:-1]``.
// - ``b``: ``(*, n, k)``, ``(n,)``, or ``LU.shape[:-1]`` (vector RHS).
// - Output: the broadcast batch followed by ``(n, k)``, or ``(n,)`` for a
//   vector RHS.
//
// Raises
// ------
// ShapeMismatch
//     When ``pivots`` is not ``LU.shape[:-1]`` or ``b`` breaks the
//     right-hand-side contract — before any LAPACK call.
// LinAlgError
//     When ``LU`` or ``b`` is not a float dtype, or when any argument is
//     null.
//
// See Also
// --------
// [[lu_factor_op]] : Produce the packed factors consumed here.
// [[solve_op]]     : Differentiable one-shot solve (factor + apply).
//
// References
// ----------
// LAPACK ``sgetrs``/``dgetrs``; *LAPACK Users' Guide* §2.5.1.
LUCID_API TensorImplPtr lu_solve_op(const TensorImplPtr& LU,
                                    const TensorImplPtr& pivots,
                                    const TensorImplPtr& b);
}  // namespace lucid
