// lucid/_C/ops/linalg/Eigh.h
//
// Symmetric / Hermitian eigendecomposition $A = V \Lambda V^\top$.
//
// Given a real symmetric (or complex Hermitian) square matrix $A$,
// computes real eigenvalues $\lambda_1 \le \cdots \le \lambda_n$
// (ascending) and orthonormal eigenvectors $V$ such that
// $A = V \mathrm{diag}(\lambda) V^\top$ with $V^\top V = I$.  Forward
// dispatches to LAPACK ``*syevd`` / ``*heevd`` on the CPU stream and to
// ``mlx::core::linalg::eigh`` (CPU stream backed) on the GPU stream.
//
// Unlike [[eig_op]], this specialised routine exploits symmetry for
// roughly $2\times$ better performance, guarantees real outputs, and
// admits a numerically stable closed-form backward.
//
// Math
// ----
// $$
//   A = V \Lambda V^\top, \qquad V^\top V = I
// $$
// The Townsend 2016 backward (when wired) reads
// $$
//   \frac{\partial L}{\partial A} = V \!\left(
//     \mathrm{diag}\!\left(\frac{\partial L}{\partial \lambda}\right)
//     + \tfrac{1}{2}\big(F \odot G + (F \odot G)^\top\big)
//   \right) V^\top
// $$
// where $G = V^\top (\partial L / \partial V)$ and
// $F_{ij} = 1 / (\lambda_j - \lambda_i)$ for $i \neq j$, $F_{ii} = 0$.
// The off-diagonal $F$ factor diverges as eigenvalues approach
// degeneracy — gradients should be regarded as unstable near a repeated
// spectrum.
//
// Notes
// -----
// - The CPU path reads only the **lower triangle** of $A$ (LAPACK
//   ``uplo = 'L'``); the strict upper triangle is ignored.  Callers
//   must ensure $A$ is genuinely symmetric — passing a non-symmetric
//   input produces meaningless results without an error.
// - Eigenvalues are guaranteed to be returned in **ascending order**.
//   Eigenvectors are real and orthonormal but defined only up to a
//   sign flip ($v_i \mapsto -v_i$ leaves $A V = V \Lambda$ unchanged);
//   the LAPACK convention pins the sign deterministically but it is
//   implementation-defined.
// - **No autograd**: the differentiable backward described in *Math*
//   above is not yet wired; outputs are leaf nodes in the autograd
//   graph.  Sign ambiguity and near-degenerate gradient blow-up are
//   the open issues blocking a default-on implementation.
//
// References
// ----------
// - Townsend, "Differentiating the Singular Value Decomposition"
//   (2016) — extends to the symmetric eigenproblem via SVD identity.
// - Anderson et al., *LAPACK Users' Guide* (3rd ed., SIAM, 1999), §2.4
//   "Symmetric Eigenproblems".
//
// See Also
// --------
// [[eig_op]]                   : General (non-symmetric) eigendecomp;
//                                slower and less stable.
// [[householder_product_op]]   : Reconstruct $Q$ from packed
//                                Householder reflectors.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute eigenvalues and eigenvectors of a real symmetric square matrix.
//
// Parameters
// ----------
// a : const TensorImplPtr&
//     Square float tensor of shape ``(..., n, n)``, interpreted as
//     symmetric.  Only the lower triangle is read; the strict upper
//     triangle is ignored.  Leading dimensions are independent batch
//     axes.  Must be at least 2-D and floating-point.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     Two-element vector ``{w, V}`` where:
//     - ``w`` has shape ``(..., n)`` — real eigenvalues in **ascending**
//       order.
//     - ``V`` has shape ``(..., n, n)`` — columns are the orthonormal
//       eigenvectors corresponding to ``w``.
//
// Math
// ----
// $A = V \mathrm{diag}(w) V^\top$ with $V^\top V = I$ and
// $w_1 \le \cdots \le w_n$.
//
// Shape
// -----
// - Input ``a``  : ``(..., n, n)``, symmetric
// - Output ``w`` : ``(..., n)``, real, ascending
// - Output ``V`` : ``(..., n, n)``, orthonormal columns
//
// Raises
// ------
// LinAlgError
//     When ``a`` is not at least 2-D, not square, or not float-typed.
//     Also when LAPACK ``*syevd`` / ``*heevd`` fails to converge
//     (reported via ``info > 0``).
//
// Notes
// -----
// Outputs are leaf nodes — autograd is **not** wired.  See header-level
// Notes for the closed-form backward and its caveats around sign
// ambiguity / spectral degeneracy.
//
// Examples
// --------
// In Python (via [[lucid.linalg.eigh]]):
// >>> A = lucid.tensor([[2.0, 1.0], [1.0, 2.0]])
// >>> w, V = lucid.linalg.eigh(A)
// >>> w
// Tensor([1., 3.])
//
// See Also
// --------
// [[eig_op]] : General non-symmetric eigendecomposition (use only
//              when the symmetric structure cannot be exploited).
LUCID_API std::vector<TensorImplPtr> eigh_op(const TensorImplPtr& a);

}  // namespace lucid
