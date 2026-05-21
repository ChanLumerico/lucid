// lucid/_C/ops/linalg/Pinv.h
//
// Moore-Penrose pseudoinverse forward op via the SVD.
//
// Given a rectangular (or singular) matrix $A \in \mathbb{R}^{m \times n}$,
// computes the unique matrix $A^+ \in \mathbb{R}^{n \times m}$ satisfying
// the four Moore-Penrose conditions
//
// $$
//   A A^+ A = A, \quad A^+ A A^+ = A^+, \quad
//   (A A^+)^\top = A A^+, \quad (A^+ A)^\top = A^+ A.
// $$
//
// The pseudoinverse generalises the ordinary inverse: when $A$ is square
// and non-singular $A^+ = A^{-1}$; when $A$ is tall and full column rank
// $A^+ = (A^\top A)^{-1} A^\top$ is the left-inverse used in least-squares.
//
// Math
// ----
// $$
//   A = U\,\Sigma\,V^\top
//   \;\Longrightarrow\;
//   A^+ = V\,\Sigma^+\,U^\top,
//   \qquad
//   \Sigma^+_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > \tau \\
//                                 0           & \text{otherwise} \end{cases}
// $$
// where $\tau$ is the rank cutoff (``rcond``) below which singular values
// are deemed numerically zero.
//
// Notes
// -----
// CPU backend computes the full SVD via LAPACK ``*gesdd``, inverts the
// non-negligible singular values, and reassembles $V \Sigma^+ U^\top$.
// GPU backend calls ``mlx::core::linalg::pinv`` (which routes through the
// CPU stream — see H3 carve-out for SVD-derived ops).
//
// The ``rcond`` threshold is chosen inside the backend following NumPy
// convention $\tau = \max(m, n) \cdot \sigma_1 \cdot \varepsilon$, where
// $\varepsilon$ is the dtype's machine epsilon.  A user-overridable
// ``rcond`` is not yet exposed at the C++ layer (the Python wrapper also
// uses the backend default).
//
// Autograd is **not** wired at the C++ level.  The Python wrapper attaches
// a backward derived from the Golub-Pereyra pseudoinverse differential:
//
// $$
//   \mathrm{d}A^+ = -A^+\,\mathrm{d}A\,A^+
//                  + A^+ A^{+\top}\,\mathrm{d}A^\top\,(I - A A^+)
//                  + (I - A^+ A)\,\mathrm{d}A^\top\,A^{+\top} A^+,
// $$
//
// which reduces to the ordinary inverse rule $\mathrm{d}A^{-1} = -A^{-1}
// \mathrm{d}A\,A^{-1}$ when $A$ is square and full rank.
//
// References
// ----------
// - Golub & Pereyra, "The Differentiation of Pseudo-Inverses and Nonlinear
//   Least Squares Problems Whose Variables Separate" (1973).
// - Penrose, "A generalized inverse for matrices" (1955).
//
// See Also
// --------
// - ``SVD.h``    — the underlying decomposition.
// - ``Lstsq.h``  — least-squares uses $A^+ b$ implicitly without forming
//   $A^+$.
// - ``Inv.h``    — ordinary matrix inverse, the square / full-rank case.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the Moore-Penrose pseudoinverse $A^+$ of a rectangular matrix.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input matrix of shape ``(..., m, n)`` with leading batch dims.  Need
//     not be square or full rank.  Must be at least 2-D and have a
//     floating-point dtype.
//
// Returns
// -------
// TensorImplPtr
//     The pseudoinverse $A^+$ of shape ``(..., n, m)`` — the last two
//     dimensions are swapped relative to the input.
//
// Shape
// -----
// - Input ``a``:  ``(..., m, n)``.
// - Output:       ``(..., n, m)``.
//
// Raises
// ------
// std::runtime_error
//     If ``a`` is null, has a non-floating-point dtype, or has fewer than
//     two dimensions.
//
// Notes
// -----
// Internally a full reduced SVD is computed; if the caller already has the
// SVD on hand, forming $A^+ = V \Sigma^+ U^\top$ manually avoids the
// redundant factorisation.
//
// Examples
// --------
// >>> // Solve a tall least-squares system via the pseudoinverse:
// >>> auto x_impl = matmul_op(pinv_op(a_impl), b_impl);
LUCID_API TensorImplPtr pinv_op(const TensorImplPtr& a);

}  // namespace lucid
