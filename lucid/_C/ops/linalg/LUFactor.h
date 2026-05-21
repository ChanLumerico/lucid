// lucid/_C/ops/linalg/LUFactor.h
//
// LU factorisation with partial pivoting — given a square matrix $A$,
// compute a permutation $P$, unit-lower-triangular $L$, and upper-
// triangular $U$ such that $PA = LU$.
//
// The result is returned in the LAPACK packed convention: a single
// $n \times n$ matrix whose upper triangle (including the diagonal)
// holds $U$ and whose strict lower triangle holds the off-diagonal
// entries of $L$ (the unit diagonal of $L$ is implicit).  Pivots are
// returned as a separate 1-based integer vector of length $n$.
//
// This op is the building block consumed by [[lu_solve_op]] and is
// the same factorisation reused internally by [[solve_op]] via
// LAPACK ``*gesv``.
//
// Math
// ----
// $$
//   P A = L U
// $$
// where $P$ is a permutation matrix encoded by the pivot indices, $L$
// is unit-lower-triangular with $\ell_{ii} = 1$, and $U$ is upper-
// triangular with $u_{ii}$ equal to the diagonal of the packed result.
//
// Notes
// -----
// - The CPU stream dispatches to LAPACK ``sgetrf_``/``dgetrf_`` via
//   ``IBackend::linalg_lu_factor()``.  The GPU stream round-trips
//   through MLX-on-CPU because Accelerate LAPACK is FP32/FP64 only.
// - Singular or numerically-ill-conditioned matrices return a packed
//   $U$ with a zero on its diagonal; downstream solves detect this
//   condition.
// - **No autograd**: the differentiable LU backward is not wired.
//   For a differentiable linear solve use [[solve_op]] directly, which
//   provides ``SolveBackward`` without exposing the LU factors.
//
// References
// ----------
// Anderson et al., *LAPACK Users' Guide* (3rd ed., SIAM, 1999), §2.5
// "LU Factorization".

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the LU factorisation $PA = LU$ of a square matrix.
//
// Parameters
// ----------
// a : const TensorImplPtr&
//     Square float tensor of shape ``(..., n, n)``.  Leading dimensions
//     are treated as independent batch axes — one factorisation is
//     produced per batch element.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     A two-element vector ``{LU_packed, pivots}``:
//     - ``LU_packed`` has the same shape and dtype as ``a``; its upper
//       triangle holds $U$ and its strict lower triangle holds the
//       off-diagonal entries of $L$ (unit diagonal is implicit).
//     - ``pivots`` has shape ``(..., n)`` and dtype ``I32``; entries
//       are 1-based row indices matching the LAPACK convention.
//
// Shape
// -----
// - Input ``a``: ``(..., n, n)``.
// - Output ``LU_packed``: ``(..., n, n)``.
// - Output ``pivots``: ``(..., n)`` with ``Dtype::I32``.
//
// Raises
// ------
// LinAlgError
//     When ``a`` is not square, not float-typed, or the LAPACK call
//     reports a structural failure.
//
// See Also
// --------
// [[lu_solve_op]] : Apply the factors to solve $Ax = b$.
// [[solve_op]]    : End-to-end differentiable linear solve.
//
// References
// ----------
// LAPACK ``sgetrf``/``dgetrf``; *LAPACK Users' Guide* §2.5.
LUCID_API std::vector<TensorImplPtr> lu_factor_op(const TensorImplPtr& a);

}  // namespace lucid
