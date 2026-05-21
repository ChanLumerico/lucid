// lucid/_C/ops/linalg/Eig.h
//
// General (non-symmetric) eigendecomposition $A = V \Lambda V^{-1}$.
//
// Given a square matrix $A$, computes eigenvalues $\lambda_i$ and right
// eigenvectors $v_i$ satisfying $A v_i = \lambda_i v_i$, packed as
// $\Lambda = \mathrm{diag}(\lambda)$ and $V = [v_1 \mid \cdots \mid v_n]$.
// Forward dispatches to LAPACK ``*geev`` on the CPU stream and to
// ``mlx::core::linalg::eig`` (CPU stream backed) on the GPU stream.
//
// Math
// ----
// $$
//   A V = V \Lambda, \qquad A = V \Lambda V^{-1}
// $$
// For a real non-symmetric $A$ both $\lambda$ and $V$ may be complex
// (occurring in conjugate pairs).  The current binding returns only the
// real part of each output — this is correct only when $A$ has a real
// spectrum (e.g. real-symmetric or positive-definite cases, for which
// [[eigh_op]] is the preferred specialised path).
//
// Notes
// -----
// - **No autograd**: a differentiable backward is not wired here.  The
//   Giles 2008 §3.1 formula
//   $$
//     \frac{\partial L}{\partial A} = V^{-\top} \!\left(
//       E \odot V^\top \frac{\partial L}{\partial V}
//       + \mathrm{diag}\!\left(\frac{\partial L}{\partial \lambda}\right)
//     \right) V^\top,
//     \quad
//     E_{ij} = \begin{cases} 1/(\lambda_j - \lambda_i) & i \neq j \\ 0 & i = j \end{cases}
//   $$
//   is numerically ill-conditioned near degenerate eigenvalues (the
//   off-diagonal $E_{ij}$ blows up when $\lambda_i \approx \lambda_j$)
//   and requires complex arithmetic in the general case.  For
//   differentiation prefer [[eigh_op]] when $A$ is symmetric.
// - Output ordering is whatever LAPACK ``*geev`` produces — there is
//   **no canonical sort** of the eigenpairs.  If a stable order is
//   required, sort by ``|w|`` or ``Re(w)`` on the caller side.
// - Eigenvectors are defined only up to a non-zero scalar; the LAPACK
//   convention normalises each column to unit Euclidean length but the
//   sign / complex phase is not pinned.
//
// References
// ----------
// - Anderson et al., *LAPACK Users' Guide* (3rd ed., SIAM, 1999), §2.4.
// - Giles, "An extended collection of matrix derivative results for
//   forward and reverse mode automatic differentiation" (2008), §3.1.
//
// See Also
// --------
// [[eigh_op]]                  : Faster, autograd-friendly path for
//                                symmetric / Hermitian $A$.
// [[householder_product_op]]   : Build $Q$ from packed Householder
//                                reflectors (used by QR-based solvers).

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute eigenvalues and right eigenvectors of a square float matrix.
//
// Parameters
// ----------
// a : const TensorImplPtr&
//     Square float tensor of shape ``(..., n, n)``.  Leading dimensions
//     are treated as independent batch axes — one decomposition per
//     batch element.  Must be at least 2-D and floating-point.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     Two-element vector ``{w, V}`` where:
//     - ``w`` has shape ``(..., n)`` — one eigenvalue per column of $A$
//       (real part only; see Notes on complex spectra).
//     - ``V`` has shape ``(..., n, n)`` — columns are the right
//       eigenvectors, normalised to unit Euclidean length.
//
// Math
// ----
// Returns $(w, V)$ such that $A V = V \mathrm{diag}(w)$, equivalently
// $A v_i = w_i v_i$ for each column $v_i$ of $V$.
//
// Shape
// -----
// - Input ``a``         : ``(..., n, n)``
// - Output ``w``        : ``(..., n)``
// - Output ``V``        : ``(..., n, n)``
//
// Raises
// ------
// LinAlgError
//     When ``a`` is not at least 2-D, not square, or not float-typed.
//     Also if LAPACK fails to converge (rare; reported via ``info > 0``).
//
// Notes
// -----
// Outputs are leaf nodes — autograd is **not** wired.  See header-level
// Notes for the deferred backward formula and its caveats.
//
// Examples
// --------
// In Python (via [[lucid.linalg.eig]]):
// >>> A = lucid.tensor([[2.0, 0.0], [0.0, 3.0]])
// >>> w, V = lucid.linalg.eig(A)
// >>> w
// Tensor([2., 3.])
//
// See Also
// --------
// [[eigh_op]] : Symmetric / Hermitian specialisation (preferred when
//               applicable — faster, real outputs, autograd-friendly).
LUCID_API std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a);

}  // namespace lucid
