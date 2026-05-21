// lucid/_C/ops/linalg/LDLFactor.h
//
// LDL$^\top$ factorisation op: decompose a symmetric (possibly indefinite)
// matrix $A$ into $A = L D L^\top$, where $L$ is unit lower-triangular and
// $D$ is block-diagonal with 1$\times$1 and 2$\times$2 blocks.
//
// Unlike Cholesky, which requires $A$ to be positive definite, LDL$^\top$
// works for any real symmetric $A$ — including indefinite or
// semi-definite matrices — because the 2$\times$2 blocks in $D$ absorb the
// directions of negative or zero curvature.  In the literature this is also
// written as $A = LBL^\top$ ($B$ for "block-diagonal").
//
// Forward dispatches to ``IBackend::linalg_ldl_factor``, which calls LAPACK
// ``*sytrf`` (``ssytrf`` / ``dsytrf``) on the CPU path using the Bunch-Kaufman
// diagonal-pivoting strategy.  The GPU path falls back to the CPU kernel
// because MLX does not expose ``*sytrf`` natively.
//
// Output packing follows the LAPACK convention:
// - The packed factor matrix has the same shape as the input ``(..., N, N)``.
//   Its strict lower triangle holds $L$ (with implicit unit diagonal) and
//   its diagonal holds the diagonal of $D$.  2$\times$2 pivot blocks in $D$
//   occupy two adjacent diagonal entries plus the corresponding sub-diagonal
//   entry of the packed factor.
// - The pivot tensor ``piv`` has shape ``(..., N)``, dtype ``I32``, encoded
//   with LAPACK's ``ipiv`` convention: positive entries mark 1$\times$1
//   pivots; equal negative entries in two consecutive rows mark a single
//   2$\times$2 pivot block.
//
// Use cases
// ---------
// Symmetric indefinite systems arise in saddle-point problems (KKT systems
// from constrained optimisation, equality-constrained least squares,
// Lagrangian Hessians).  LDL$^\top$ is the canonical direct method there;
// Cholesky would fail because $A$ is not SPD.
//
// References
// ----------
// - Bunch & Kaufman, "Some stable methods for calculating inertia and
//   solving symmetric linear systems" (1977).
// - LAPACK Users' Guide §2.4.2, "Symmetric Indefinite".

#pragma once
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the LDL$^\top$ factorisation of a symmetric matrix.
//
// Returns the packed factor and the pivot index tensor.  Both follow the
// LAPACK ``*sytrf`` storage convention so they are consumable by
// ``lucid.linalg.ldl_solve`` (which composes triangular solves).  Batched
// inputs are supported: leading dimensions are treated as independent
// matrices and the factorisation is run on each $N \times N$ slice.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Square symmetric matrix of shape ``(..., N, N)`` with dtype ``F32`` or
//     ``F64``.  The matrix need not be positive definite; only the lower
//     triangle is read.
//
// Returns
// -------
// std::vector<TensorImplPtr> of size 2
//     - ``[0]`` packed factor ``LD`` of shape ``(..., N, N)``, same dtype as
//       ``a``.  Strict lower triangle stores $L$ (unit diagonal implicit);
//       diagonal and sub-diagonal entries jointly encode $D$ including
//       2$\times$2 pivot blocks.
//     - ``[1]`` pivot tensor of shape ``(..., N)``, dtype ``I32``, following
//       the LAPACK ``ipiv`` convention (positive: 1$\times$1 pivot; pair of
//       equal negatives: 2$\times$2 pivot block).
//
// Math
// ----
// $$
//   P A P^\top = L D L^\top,
// $$
// where $P$ is the permutation implied by the pivots, $L$ is unit
// lower-triangular, and $D$ is block-diagonal with 1$\times$1 and
// 2$\times$2 blocks.  The 2$\times$2 blocks are necessary when no
// scalar pivot can be chosen without losing numerical stability — this is
// what allows the factorisation to handle indefinite $A$.
//
// Shape
// -----
// - ``a``           : ``(..., N, N)``.
// - return ``[0]``  : ``(..., N, N)``.
// - return ``[1]``  : ``(..., N)``.
//
// Raises
// ------
// LucidError
//     If ``a`` is not at least 2-D, not square, or has a non-float dtype.
// LucidError
//     If LAPACK ``info > 0`` — $D$ is exactly singular, so $A$ is rank
//     deficient and the factorisation cannot be completed.
//
// Notes
// -----
// - No autograd node is registered; the outputs are leaves.
// - The current ``ldl_solve`` wrapper in ``lucid.linalg`` only supports
//   1$\times$1 pivots (no 2$\times$2 Bunch-Kaufman block solving yet).
//
// See Also
// --------
// - ``cholesky_op`` — cheaper alternative when $A$ is SPD.
// - ``solve_triangular_op`` — building block for ``ldl_solve``.
LUCID_API std::vector<TensorImplPtr> ldl_factor_op(const TensorImplPtr& a);

}  // namespace lucid
