// lucid/_C/ops/linalg/QR.h
//
// Reduced QR decomposition $A = QR$ for batched float matrices.
//
// Forward factorises an ``(..., m, n)`` matrix into an orthonormal
// ``Q`` and an upper-triangular ``R``.  The "reduced" (a.k.a. "thin"
// / "economy") variant is returned: ``Q`` has shape ``(..., m, k)``
// and ``R`` has shape ``(..., k, n)`` with $k = \min(m, n)$.  This is
// the form most useful in ML pipelines (least squares, Gram-Schmidt
// orthonormalisation) and avoids materialising the orthogonal
// complement of the column space.
//
// CPU dispatch is LAPACK's ``*geqrf`` (Householder reflectors packed
// into the lower triangle of the input) followed by ``*orgqr`` (which
// materialises $Q$ explicitly).  GPU dispatch is
// ``mlx::core::linalg::qr`` executed on the CPU stream (the MLX-on-CPU
// linalg carve-out documented in DEVELOPMENT.md §H3).
//
// Autograd is intentionally not wired: a future backward will use
// Walter & Lehmann's formula (or the equivalent Bettale 2013 form):
// $$
//   M = R^\top \bar R - \bar Q^\top Q, \qquad
//   \Phi(M) = \mathrm{tril}(M) - \tfrac{1}{2}\mathrm{tril}(M)^\top,
// $$
// $$
//   \frac{\partial L}{\partial A} =
//     \bigl(\bar Q + Q\,\Phi(R^\top \bar R - \bar Q^\top Q)\bigr) R^{-\top}.
// $$
// This requires a batched triangular solve and is deferred.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the reduced QR factorisation of a float matrix.
//
// Forward calls LAPACK ``*geqrf`` + ``*orgqr`` on the CPU stream or
// ``mlx::core::linalg::qr`` on the GPU stream and returns the two
// factors as a length-2 vector ``{Q, R}``.  Autograd is not wired —
// both outputs are leaves in the gradient graph.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input matrix of shape ``(..., m, n)``.  Must be at least 2-D
//     and have a floating-point dtype.  Leading dimensions are
//     treated as independent batch dimensions.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     Length-2 vector ``{Q, R}`` where
//
//     - ``Q`` has shape ``(..., m, k)`` with $Q^\top Q = I_k$,
//     - ``R`` has shape ``(..., k, n)`` and is upper-triangular,
//
//     and $k = \min(m, n)$.
//
// Math
// ----
// $$
//   A = Q R, \qquad Q^\top Q = I_k, \qquad R = \mathrm{triu}(R).
// $$
//
// Shape
// -----
// - ``a``: ``(..., m, n)``.
// - ``Q``: ``(..., m, k)``, ``R``: ``(..., k, n)`` with
//   $k = \min(m, n)$.
//
// Notes
// -----
// - Only the reduced ("thin") factorisation is produced; ``mode='complete'``
//   parity with the reference framework is not currently exposed.
// - No backward is registered.  Wrapping ``qr_op`` inside a graph
//   where ``Q`` or ``R`` requires gradients will yield leaf tensors.
//
// Raises
// ------
// ValueError
//     If ``a`` is fewer than 2 dimensions or has a non-float dtype.
//
// References
// ----------
// Golub & Van Loan, *Matrix Computations* (4th ed.), Algorithm 5.2.1
// (Householder QR).
// Walter & Lehmann, "Algorithmic Differentiation of Linear Algebra
// Functions" (2010), §3.
// Bettale, "Differentiating the QR Decomposition" (2013).
//
// See Also
// --------
// inv_op : Matrix inverse, often composed with QR for least squares.
// Lstsq.h : Direct least-squares solver built on top of QR.
LUCID_API std::vector<TensorImplPtr> qr_op(const TensorImplPtr& a);

}  // namespace lucid
