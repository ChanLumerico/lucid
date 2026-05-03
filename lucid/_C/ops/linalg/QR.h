// lucid/_C/ops/linalg/QR.h
//
// QR decomposition op: given an (m×n) matrix A, compute the reduced QR
// factorisation A = Q R where Q is (m×k) with orthonormal columns and R is
// (k×n) upper-triangular, with k = min(m, n).
//
// The "reduced" (also called "thin" or "economy") factorisation is returned
// rather than the full Q because it is the form most commonly needed in ML:
//   - For m >= n: Q is (m×n), R is (n×n).
//   - For m <  n: Q is (m×m), R is (m×n).
//
// Forward dispatch goes to IBackend::linalg_qr(), which uses LAPACK's
// dgeqrf (computes the Householder reflectors in a packed representation)
// followed by dorgqr (materialises Q) on the CPU path, and
// mlx::core::linalg::qr on the GPU path.
// Returns {Q, R} as a vector of two TensorImplPtrs.
//
// Note: no backward node is registered.  A future backward would implement
// Luk Bettale's formula (from "Differentiating the QR Decomposition", 2013):
//   Given G_Q = ∂L/∂Q and G_R = ∂L/∂R:
//     M = Rᵀ G_R - G_Qᵀ Q            (antisymmetric part drives dA)
//     Phi(M) = tril(M) - tril(M)ᵀ/2   (extract the skew-symmetric component)
//     dA = (Q G_R + Q Phi(Rᵀ G_R - G_Qᵀ Q)) Rᵀ⁻¹
//   This requires a triangular solve and is deferred to a future phase.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the reduced QR factorisation of matrix a.
//
// Returns {Q, R} where:
//   Q has shape (..., m, k) — columns are orthonormal: Qᵀ Q = I_k
//   R has shape (..., k, n) — upper-triangular
// with k = min(m, n).  Validates that a is at least 2-D and float-typed.
// Batched inputs are supported; the batch dimensions are all but the last two.
// Autograd is not wired; the output tensors are leaves in the gradient graph.
LUCID_API std::vector<TensorImplPtr> qr_op(const TensorImplPtr& a);

}  // namespace lucid
