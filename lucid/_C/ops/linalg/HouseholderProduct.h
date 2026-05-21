// lucid/_C/ops/linalg/HouseholderProduct.h
//
// Reconstruct the orthogonal factor $Q$ from a packed sequence of
// Householder reflectors.
//
// Given the packed reflector matrix $H$ and scaling factors $\tau$
// returned by QR-type factorisations (e.g. LAPACK ``*geqrf``,
// Hessenberg / tridiagonal reductions, or the eigenvector synthesis
// step of [[eig_op]] / [[eigh_op]]), forms the explicit orthogonal
// matrix
// $$
//   Q = H_1 H_2 \cdots H_k,
//   \qquad
//   H_i = I - \tau_i\, v_i v_i^\top
// $$
// where each reflector $v_i$ is read from column $i$ of $H$ (the strict
// lower triangle below the diagonal, with an implicit unit on the
// diagonal — the LAPACK convention).
//
// Forward dispatches to LAPACK ``*orgqr`` on the CPU stream and the
// MLX equivalent on the GPU stream.  No backward is wired: this op is
// a building block consumed by higher-level differentiable routines
// (notably QR-based solvers), and its inputs are themselves outputs of
// non-differentiable factorisations.
//
// Math
// ----
// $$
//   Q = \prod_{i=1}^{k} \big(I - \tau_i\, v_i v_i^\top\big),
//   \qquad k = \min(m, n)
// $$
// with each $v_i \in \mathbb{R}^m$ satisfying $v_i[1{:}i-1] = 0$ and
// $v_i[i] = 1$ (implicit).  The resulting $Q$ has orthonormal columns
// ($Q^\top Q = I_k$) and forms the thin / economy $Q$ factor.
//
// Notes
// -----
// - This is the standard LAPACK "unpack" step: $H$ + $\tau$ is the
//   compact storage produced by ``*geqrf``, and ``*orgqr`` materialises
//   $Q$ in $O(m k^2)$ operations.
// - **No autograd**: derivatives flow through the higher-level QR /
//   eig op that owns this reconstruction.
// - The current implementation does **not** batch over leading
//   dimensions — only the trailing two axes of $H$ are read.
//
// References
// ----------
// - Anderson et al., *LAPACK Users' Guide* (3rd ed., SIAM, 1999) —
//   ``*orgqr`` "Generate Q from QR factorisation".
// - Golub & Van Loan, *Matrix Computations* (4th ed., 2013), §5.1.2
//   "Householder Reflections".

#pragma once
#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Materialise the orthogonal $Q$ from packed Householder reflectors.
//
// Parameters
// ----------
// H : const TensorImplPtr&
//     Packed reflector matrix of shape ``(m, n)`` produced by a QR-type
//     factorisation.  Column $i$ stores reflector $v_i$ in the strict
//     lower triangle below the diagonal (the diagonal entry of $v_i$
//     is implicitly $1$ and the slot is reused for $R$).  Must be
//     float-typed and non-null.
// tau : const TensorImplPtr&
//     1-D scaling factors of length $k = \min(m, n)$.  ``tau[i]`` is
//     the $\tau_i$ scalar of the $i$-th reflector
//     $H_i = I - \tau_i v_i v_i^\top$.  Must be float-typed and
//     non-null.
//
// Returns
// -------
// TensorImplPtr
//     Explicit orthogonal matrix $Q$ of shape ``(m, k)`` with the same
//     dtype as ``H``.  Columns are orthonormal ($Q^\top Q = I_k$).
//
// Math
// ----
// $$
//   Q = H_1 H_2 \cdots H_k, \qquad k = \min(m, n)
// $$
//
// Shape
// -----
// - Input ``H``    : ``(m, n)``
// - Input ``tau``  : ``(k,)`` with $k = \min(m, n)$
// - Output ``Q``   : ``(m, k)``
//
// Raises
// ------
// LinAlgError
//     When ``H`` or ``tau`` is null / not float-typed.
//
// Notes
// -----
// Output is a leaf node — autograd is **not** wired.  Gradients flow
// through the calling QR / eig op rather than through this unpack
// step.
//
// See Also
// --------
// [[eig_op]]  : Produces packed reflectors for general eigenvector
//               reconstruction.
// [[eigh_op]] : Symmetric eigendecomp; reflectors come from a
//               tridiagonal reduction step.
//
// References
// ----------
// LAPACK ``*orgqr``; Golub & Van Loan §5.1.2.
LUCID_API TensorImplPtr householder_product_op(const TensorImplPtr& H, const TensorImplPtr& tau);
}  // namespace lucid
