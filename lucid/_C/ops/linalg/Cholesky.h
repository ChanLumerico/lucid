// lucid/_C/ops/linalg/Cholesky.h
//
// Cholesky decomposition op: given a symmetric positive-definite (SPD) matrix
// A, compute its lower-triangular factor L such that A = L Lᵀ.  When
// upper=true, returns the upper-triangular factor U such that A = Uᵀ U.
//
// Forward dispatch goes to IBackend::linalg_cholesky(), which uses LAPACK's
// dpotrf (Cholesky factorisation, reading only the lower or upper triangle
// of A) on the CPU path, and mlx::core::linalg::cholesky on the GPU path.
// The output has the same shape as the input; the unused triangle is zeroed.
//
// Intended use: Cholesky decomposition is frequently used in probabilistic ML
// for sampling from multivariate Gaussians (sample = L z where z ~ N(0, I))
// and for stable evaluation of log-determinants (log det A = 2 sum_i log L_ii).
//
// Note: this op currently has no registered backward node, so gradients
// cannot flow through it.  The Cholesky backward (Iain Murray's formula,
// as published in "Differentiation of the Cholesky decomposition", 2016)
// should be added in a future phase.  Murray's formula solves a symmetric
// triangular system: dA = L⁻¹ Phi(Lᵀ G) L⁻ᵀ  (suitably symmetrised)
// where G = ∂L/∂L is the upstream gradient and Phi zeros the strictly
// upper-triangular part.
//
// Precondition: A must be symmetric positive-definite.  The backend does not
// check this; passing a non-SPD matrix results in LAPACK returning info > 0
// (factorisation failed) which is translated to a LucidError.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the Cholesky factor of a symmetric positive-definite matrix A.
//
// When upper=false (the default), returns the lower-triangular L such that
//   A = L Lᵀ.
// When upper=true, returns the upper-triangular U such that A = Uᵀ U.
// Validates that a is at least 2-D, square, and float-typed before dispatching.
// Batched inputs are supported: all dimensions except the last two are batch.
LUCID_API TensorImplPtr cholesky_op(const TensorImplPtr& a, bool upper = false);

}  // namespace lucid
