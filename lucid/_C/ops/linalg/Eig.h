// lucid/_C/ops/linalg/Eig.h
//
// General (non-symmetric) eigendecomposition op: given a square matrix A,
// compute eigenvalues w and eigenvectors V such that A V = V diag(w).
//
// Forward dispatch goes to IBackend::linalg_eig(), which uses LAPACK's
// dgeev on the CPU path and mlx::core::linalg::eig on the GPU path.
// Returns {w, V} where:
//   w has shape (..., n)    — one eigenvalue per column
//   V has shape (..., n, n) — columns are the right eigenvectors
//
// Note: for general (non-symmetric) matrices the eigenvalues and eigenvectors
// may be complex.  The current implementation returns real types; callers
// should ensure A is real-symmetric if they require real results.
//
// Note: no backward node is registered.  The backward for the general
// eigendecomposition is numerically delicate (complex arithmetic, degenerate
// eigenvalues) and is deferred to a future phase.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute eigenvalues and right eigenvectors of square float matrix a.
//
// Returns {w, V} where w has shape (..., n) and V has shape (..., n, n).
// Validates that a is at least 2-D, square, and float-typed before dispatching.
LUCID_API std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a);

}  // namespace lucid
