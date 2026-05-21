// lucid/_C/ops/linalg/MatrixPower.h
//
// Integer matrix power $A^n$ for square float matrices and
// $n \in \mathbb{Z}$.
//
// The backend uses repeated squaring (binary exponentiation), which
// performs $O(\log |n|)$ matrix multiplies instead of the naive
// $O(|n|)$ chain.  Special cases:
//
// - $n = 0$ $\to$ identity matrix $I$ of the same size as $A$.
// - $n > 0$ $\to$ standard repeated squaring of $A$.
// - $n < 0$ $\to$ $(A^{-1})^{|n|}$; requires $A$ to be invertible.
//
// Forward dispatch goes to ``IBackend::linalg_matrix_power``.  CPU
// path uses CBLAS ``gemm`` for the squarings (and LAPACK
// ``*getrf`` / ``*getri`` for the inverse when $n < 0$); the GPU
// path uses MLX matmul with the same exponentiation logic.
//
// Autograd is intentionally not wired.  A future backward would
// differentiate the binary-decomposition tree: for a chain of $k$
// squarings the gradient is a sum of $k$ terms each involving the
// intermediate powers, giving the product-rule expansion
// $$
//   \frac{\partial L}{\partial A}
//     = \sum_{i=0}^{n-1} (A^i)^\top
//       \,\frac{\partial L}{\partial A^n}\,(A^{n-1-i})^\top.
// $$
// This is deferred to a future phase.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute $A^n$ for a square float matrix and integer exponent.
//
// Validates that ``a`` is at least 2-D, square in the trailing two
// axes, and float-typed; dispatches to the backend.  The output
// shape always matches ``a``'s shape (matrix powers preserve shape).
// Autograd is not wired — the result is a leaf in the gradient
// graph.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input matrix of shape ``(..., N, N)``.
// n : int
//     Integer exponent.  Named ``n`` for parity with the reference
//     framework's ``matrix_power(a, n)`` signature; the algorithm
//     refers to it as $p$ internally.  May be zero, positive, or
//     negative; negative values require $A$ to be invertible.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``(..., N, N)`` holding $A^n$ (and the
//     identity matrix if ``n == 0``).
//
// Math
// ----
// $$
//   A^n = \begin{cases}
//     I & n = 0, \\
//     \underbrace{A \cdot A \cdots A}_{n\;\text{times}} & n > 0, \\
//     (A^{-1})^{|n|} & n < 0.
//   \end{cases}
// $$
// Repeated squaring decomposes $n = \sum_i b_i 2^i$ (its binary
// representation) and accumulates the result in
// $O(\log |n|)$ matrix multiplies.
//
// Shape
// -----
// - ``a``: ``(..., N, N)``.
// - Output: same shape as ``a``.
//
// Notes
// -----
// - No backward is registered; ``matrix_power`` outputs are leaves.
// - For graph neural networks $A$ is often an adjacency or
//   normalised Laplacian — large ``n`` is fine because cost grows
//   only logarithmically.
//
// Raises
// ------
// ValueError
//     If ``a`` is not square, fewer than 2-D, or non-float.
// LinAlgError
//     (At backend level, when ``n < 0``.) If $A$ is singular.
//
// References
// ----------
// Knuth, *The Art of Computer Programming, Vol. 2* §4.6.3
// (right-to-left binary exponentiation).
// Higham, *Functions of Matrices* (2008), §1.2.
//
// See Also
// --------
// inv_op : Used internally when ``n < 0``.
// matmul_op : Single-step underlying primitive.
LUCID_API TensorImplPtr matrix_power_op(const TensorImplPtr& a, int n);

}  // namespace lucid
