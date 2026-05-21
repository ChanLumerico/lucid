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
// - Multi-RHS solves are supported by passing ``b`` with shape
//   ``(..., n, k)``; the same factors are applied to all $k$ columns.
// - **No autograd**: differentiable solves should use [[solve_op]]
//   instead — that op factorises internally and wires the backward
//   identities $\partial L / \partial b = A^{-\top}\, \partial L /
//   \partial x$ and $\partial L / \partial A = -(\partial L /
//   \partial b)\, x^\top$.
//
// References
// ----------
// Anderson et al., *LAPACK Users' Guide* (3rd ed., SIAM, 1999),
// §2.5.1 "Solving Linear Systems".

#pragma once
#include "../../api.h"
#include "../../core/fwd.h"
namespace lucid {

// Solve $Ax = b$ given the packed LU factors and pivot vector.
//
// Parameters
// ----------
// LU : const TensorImplPtr&
//     Packed LU matrix of shape ``(..., n, n)`` as returned by
//     [[lu_factor_op]] — upper triangle holds $U$, strict lower
//     triangle holds the off-diagonal entries of $L$.
// pivots : const TensorImplPtr&
//     1-based pivot indices of shape ``(..., n)`` and dtype ``I32``.
// b : const TensorImplPtr&
//     Right-hand side of shape ``(..., n)`` for a single RHS or
//     ``(..., n, k)`` for $k$ simultaneous RHS columns.
//
// Returns
// -------
// TensorImplPtr
//     Solution tensor with the same shape and dtype as ``b``.
//
// Shape
// -----
// - ``LU``: ``(..., n, n)``.
// - ``pivots``: ``(..., n)``.
// - ``b``: ``(..., n)`` or ``(..., n, k)``.
// - Output: same shape as ``b``.
//
// Raises
// ------
// LinAlgError
//     When ``LU`` or ``b`` is not a float dtype, when any argument is
//     null, or when shapes are inconsistent with the batched solve.
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
