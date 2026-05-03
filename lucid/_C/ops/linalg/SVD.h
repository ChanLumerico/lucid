// lucid/_C/ops/linalg/SVD.h
//
// Singular Value Decomposition op: given a matrix A of shape (..., m, n),
// compute the reduced SVD:
//   A = U diag(S) Vhᵀ   (note: Vh is Vᵀ, not V)
// where:
//   U  has shape (..., m, k) — left singular vectors (columns are orthonormal)
//   S  has shape (..., k)    — singular values in descending order
//   Vh has shape (..., k, n) — right singular vectors (rows are orthonormal)
// with k = min(m, n).
//
// The "reduced" (economy) SVD is returned; the full SVD padding to min/max
// shapes is not currently exposed.
//
// Forward dispatch goes to IBackend::linalg_svd(), which uses LAPACK's
// dgesdd (divide-and-conquer SVD) on the CPU path, and
// mlx::core::linalg::svd on the GPU path.
//
// When compute_uv=true (the default), returns {U, S, Vh}.
// When compute_uv=false, returns {S} only (more efficient when only singular
// values are needed).
//
// Note: no backward node is registered.  A future backward would implement
// the Papadopoulo-Lourakis formula (from "Estimating the Jacobian of the SVD",
// 2000):
//   ∂L/∂A = U (K ⊙ (Uᵀ G_U)) Vhᵀ + U diag(G_S) Vhᵀ + U (K' ⊙ (Vhᵀ G_Vh)) Vhᵀ
// where K is a matrix of 1/(sᵢ² - sⱼ²) factors and G_U, G_S, G_Vh are the
// upstream gradients.  The formula must handle degenerate singular values.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the SVD of matrix a.
//
// When compute_uv=true (default), returns {U, S, Vh} where:
//   U   has shape (..., m, k)  — left singular vectors
//   S   has shape (..., k)     — singular values, descending
//   Vh  has shape (..., k, n)  — right singular vectors (transposed)
// When compute_uv=false, returns {S} only (skips materialising U and Vh).
// Validates that a is at least 2-D and float-typed.
// Autograd is not wired; outputs are leaf nodes in the gradient graph.
LUCID_API std::vector<TensorImplPtr> svd_op(const TensorImplPtr& a, bool compute_uv = true);

}  // namespace lucid
