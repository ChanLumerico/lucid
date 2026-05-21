// lucid/_C/ops/linalg/Cholesky.h
//
// Cholesky decomposition op: factor a symmetric positive-definite matrix $A$
// into a triangular product $A = LL^\top$ (or $A = U^\top U$).
//
// The forward kernel reads only one triangle of $A$ and writes the
// corresponding triangular factor in-place; the opposite triangle of the
// output is zeroed by the backend.  On the CPU path the work is delegated to
// LAPACK ``*potrf`` (``spotrf`` / ``dpotrf``); on the GPU path to
// ``mlx::core::linalg::cholesky``.  Both paths use the linalg stream
// constant in ``_Detail.h``.
//
// Cholesky is one of the most-used linear-algebra primitives in probabilistic
// ML: sampling $x = L z$ with $z \sim \mathcal{N}(0, I)$ produces correlated
// Gaussian samples, and $\log\det A = 2 \sum_i \log L_{ii}$ is the numerically
// stable form of the multivariate-normal log-density.
//
// Notes
// -----
// - No backward node is registered at the engine level.  The autograd-aware
//   forward (Murray 2016 formula, see ``lucid/linalg/__init__.py``) lives in
//   the Python wrapper, which composes ``solve_triangular`` calls.
// - The input is *not* checked for SPD-ness; that information only comes back
//   as a non-zero LAPACK ``info`` from ``*potrf``.
//
// References
// ----------
// - Murray, "Differentiation of the Cholesky decomposition" (2016).

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the Cholesky factor of a symmetric positive-definite matrix.
//
// Returns the lower-triangular $L$ (default) or upper-triangular $U$ such
// that $A = LL^\top$ or $A = U^\top U$ respectively.  Forward dispatches to
// LAPACK ``*potrf`` (CPU) or ``mlx::core::linalg::cholesky`` (GPU).  Batched
// inputs are supported: every dimension except the trailing two is treated
// as a batch axis and the factorisation is run independently on each
// $N \times N$ slice.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Square symmetric positive-definite matrix of shape ``(..., N, N)``,
//     dtype ``F32`` or ``F64``.
// upper : bool, optional
//     If ``true`` return the upper factor $U$; otherwise (default) return
//     the lower factor $L$.
//
// Returns
// -------
// TensorImplPtr
//     Triangular Cholesky factor with the same shape and dtype as ``a``.
//     The opposite triangle is filled with zeros.
//
// Math
// ----
// $$
//   A = LL^\top \quad\text{or}\quad A = U^\top U.
// $$
// The corresponding reverse-mode rule (Murray 2016) is
// $$
//   \frac{\partial L}{\partial A} = \tfrac{1}{2}\, L^{-\top}
//     \Phi\!\bigl(L^\top \tfrac{\partial L}{\partial L}\bigr) L^{-1},
// $$
// where $\Phi(M) = \mathrm{tril}(M) + \mathrm{tril}(M, -1)^\top$ copies the
// strict lower triangle into the strict upper.  This formula is realised in
// the Python wrapper via two triangular solves, not in this engine op.
//
// Shape
// -----
// - ``a``: ``(..., N, N)``.
// - return: ``(..., N, N)``.
//
// Raises
// ------
// LucidError
//     If ``a`` is not at least 2-D, not square, or has a non-float dtype.
// LucidError
//     If LAPACK ``info > 0`` — a non-positive pivot was encountered, i.e.
//     ``a`` is not positive definite (within working precision).
//
// Notes
// -----
// No autograd node is wired; the returned ``TensorImpl`` has
// ``requires_grad = false``.  Gradient-aware Cholesky is provided one layer
// up by ``lucid.linalg.cholesky``.
//
// See Also
// --------
// - ``solve_triangular_op`` — triangular back/forward substitution used in
//   the Python-side Cholesky backward.
// - ``ldl_factor_op`` — analogous factorisation for symmetric indefinite
//   matrices.
LUCID_API TensorImplPtr cholesky_op(const TensorImplPtr& a, bool upper = false);

}  // namespace lucid
