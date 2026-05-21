// lucid/_C/ops/linalg/SolveTriangular.h
//
// Triangular linear-system solve: given a triangular matrix $A$ and a
// right-hand-side $B$, compute $X$ such that $A X = B$.
//
// This is strictly cheaper than the general ``solve_op``: no factorisation
// is performed — LAPACK ``*trtrs`` (``strtrs`` / ``dtrtrs``) does a single
// pass of forward- or back-substitution, $\mathcal{O}(n^2)$ per right-hand
// side instead of $\mathcal{O}(n^3)$ for a Gaussian-elimination solve.  This
// is exactly the inner kernel used to back-substitute through a Cholesky,
// QR, or LDL$^\top$ factor.
//
// The forward kernel only reads the relevant triangle of $A$:
// - ``upper=true``  : the strict lower triangle of $A$ is ignored.
// - ``upper=false`` : the strict upper triangle of $A$ is ignored.
// - ``unitriangular=true`` : the diagonal of $A$ is treated as all-ones and
//   the stored diagonal entries are ignored (used when $A$ is the unit
//   lower factor returned by ``ldl_factor`` or by Householder routines).
//
// Forward dispatches to ``IBackend::linalg_solve_triangular`` → LAPACK
// ``*trtrs`` on the CPU path.  No GPU-native dispatch is wired; the GPU
// backend round-trips through the CPU LAPACK routine.
//
// Notes
// -----
// - No autograd node is registered at the engine layer.  The Python
//   ``lucid.linalg.solve_triangular`` wrapper composes manual transpose +
//   triangular-solve operations to realise the reverse-mode rule
//   $\partial B = A^{-\top}\,\partial X$ and
//   $\partial A = -A^{-\top}\,\partial X\, X^\top$.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Solve $A X = B$ for $X$ where $A$ is triangular.
//
// Performs a single substitution sweep through $A$; no factorisation is
// required.  $A$ and $B$ must share the same dtype and device.  Batched
// inputs are supported on both arguments: the trailing two dimensions are
// treated as the matrix axes and the broadcast batch loop runs LAPACK once
// per slice.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Triangular coefficient matrix of shape ``(..., N, N)`` with dtype
//     ``F32`` or ``F64``.  Only the half indicated by ``upper`` is read.
// b : TensorImplPtr
//     Right-hand side of shape ``(..., N, K)`` (or ``(..., N)`` for a single
//     RHS), same dtype and device as ``a``.
// upper : bool, optional
//     If ``true`` (default), ``a`` is interpreted as upper-triangular and
//     back-substitution is used.  If ``false``, ``a`` is lower-triangular
//     and forward-substitution is used.
// unitriangular : bool, optional
//     If ``true``, the diagonal of ``a`` is treated as all-ones regardless
//     of its stored values.  Useful for the unit-lower factor produced by
//     LDL$^\top$ and for Householder products.  Default is ``false``.
//
// Returns
// -------
// TensorImplPtr
//     Solution $X$ with the same shape and dtype as ``b``.
//
// Math
// ----
// Solves
// $$
//   A X = B \quad\Longleftrightarrow\quad X = A^{-1} B,
// $$
// without ever forming $A^{-1}$.  When ``unitriangular`` is set, $A$ is
// regarded as $\widetilde{A}$ with $\widetilde{A}_{ii} = 1$ for all $i$.
// The corresponding reverse-mode rule is
// $$
//   \frac{\partial B}{\partial L} = A^{-\top} \frac{\partial X}{\partial L},
//   \qquad
//   \frac{\partial A}{\partial L} = -\,\frac{\partial B}{\partial L}\,X^\top,
// $$
// implemented in the Python wrapper.
//
// Shape
// -----
// - ``a`` : ``(..., N, N)``.
// - ``b`` : ``(..., N, K)`` or ``(..., N)``.
// - return: same shape as ``b``.
//
// Raises
// ------
// LucidError
//     If ``a`` is not square, if ``a`` and ``b`` have mismatched dtype or
//     device, or if either tensor has a non-float dtype.
// LucidError
//     If LAPACK reports a singular triangular system (zero on the diagonal
//     when ``unitriangular`` is false).
//
// Notes
// -----
// No autograd node is wired; the output ``TensorImpl`` is a leaf.
//
// See Also
// --------
// - ``solve_op`` — full LU-based dense solve; use when $A$ is not known to
//   be triangular.
// - ``cholesky_op`` / ``ldl_factor_op`` — produce the triangular factors
//   this op back-substitutes through.
LUCID_API TensorImplPtr solve_triangular_op(const TensorImplPtr& a,
                                            const TensorImplPtr& b,
                                            bool upper = true,
                                            bool unitriangular = false);

}  // namespace lucid
