// lucid/_C/ops/linalg/Lstsq.h
//
// Linear least-squares forward op for over- or under-determined systems.
//
// Given a coefficient matrix $A \in \mathbb{R}^{m \times n}$ and a
// right-hand side $B \in \mathbb{R}^{m \times k}$ (or a vector $B \in
// \mathbb{R}^m$), finds $X$ minimising the squared residual
//
// $$
//   X = \arg\min_X \,\|A\,X - B\|_2^2.
// $$
//
// The op handles all three regimes uniformly:
// - **Over-determined** ($m > n$, $A$ full column rank): unique minimiser
//   $X = (A^\top A)^{-1} A^\top B$, computed stably via QR / SVD without
//   forming the normal equations.
// - **Square** ($m = n$, $A$ non-singular): coincides with the exact
//   solution $X = A^{-1} B$.
// - **Under-determined** ($m < n$, or rank-deficient): returns the
//   minimum-norm minimiser $X = A^+ B$.
//
// Math
// ----
// $$
//   X^\star = A^+ B,\qquad
//   A^+ = V\,\Sigma^+\,U^\top,\qquad
//   \|A X^\star - B\|_2^2 = \|(I - A A^+) B\|_2^2.
// $$
//
// Notes
// -----
// CPU backend dispatches to LAPACK's ``*gels`` (QR-based, requires $A$ to
// be full rank) for the default driver, or ``*gelsd`` (SVD-based,
// rank-revealing divide-and-conquer) when rank-deficient inputs are
// expected.  GPU backend currently routes through the CPU stream — there
// is no native Metal least-squares kernel yet (H3 carve-out shared with
// other SVD/QR-derived ops).
//
// Only the solution tensor is materialised by this op.  The Python
// wrapper's tuple slots for ``residuals`` / ``rank`` / ``singular_values``
// are populated with empty placeholders for NumPy-API compatibility — the
// engine does not yet plumb these through.
//
// Autograd is **not** wired at the C++ level.  A future backward would
// follow the normal-equation chain rule
//
// $$
//   \frac{\partial L}{\partial A}
//     = -A^{+\top}\frac{\partial L}{\partial X}\,X^\top
//       + (I - A A^+)\,\frac{\partial L}{\partial X}\,X^\top A^{+\top} A^+,
//   \qquad
//   \frac{\partial L}{\partial B} = A^{+\top}\frac{\partial L}{\partial X}.
// $$
//
// References
// ----------
// - Anderson et al., "LAPACK Users' Guide" 3rd ed. — §2.4.2 (least
//   squares), §4.1 (``*gels`` family).
// - Björck, "Numerical Methods for Least Squares Problems" (1996).
//
// See Also
// --------
// - ``Pinv.h`` — pseudoinverse; ``lstsq(A, B)`` is conceptually
//   ``pinv(A) @ B`` but avoids forming $A^+$ explicitly.
// - ``QR.h``   — the orthogonal factorisation used by the default driver.
// - ``SVD.h``  — the decomposition behind the rank-revealing driver.

#pragma once
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Solve the linear least-squares problem $\min_X \|A X - B\|_2$.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Coefficient matrix of shape ``(..., m, n)`` with leading batch dims.
//     Must be at least 2-D and have a floating-point dtype.
// b : TensorImplPtr
//     Right-hand side of shape ``(..., m, k)`` (multiple RHS) or
//     ``(..., m)`` (single RHS).  Must share dtype with ``a``.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     A one-element vector ``{solution}`` where ``solution`` has shape
//     ``(n, k)`` when ``nrhs > 1`` and ``(n,)`` when ``nrhs == 1``.  The
//     vector wrapping anticipates a future expansion to
//     ``{solution, residuals, rank, singular_values}`` matching the
//     NumPy/reference-framework signature.
//
// Shape
// -----
// - Input ``a``: ``(..., m, n)``.
// - Input ``b``: ``(..., m, k)`` or ``(..., m)``.
// - Output:      ``(n, k)`` or ``(n,)``.
//
// Raises
// ------
// std::runtime_error
//     If either input is null or has a non-floating-point dtype.
//
// Notes
// -----
// The current backend uses ``*gels`` (QR driver), which assumes $A$ is
// full rank.  Rank-deficient inputs may produce silently inaccurate
// solutions until ``*gelsd`` / ``*gelsy`` drivers are exposed.
//
// Examples
// --------
// >>> // Solve A x = b in the least-squares sense (tall A, single RHS):
// >>> auto x_impl = lstsq_op(a_impl, b_impl)[0];
LUCID_API std::vector<TensorImplPtr> lstsq_op(const TensorImplPtr& a, const TensorImplPtr& b);
}  // namespace lucid
