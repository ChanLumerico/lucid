// lucid/_C/ops/linalg/Eigh.h
//
// Symmetric/Hermitian eigendecomposition op: given a real symmetric (or
// complex Hermitian) square matrix A, compute real eigenvalues w in ascending
// order and their corresponding orthonormal eigenvectors V such that
//   A = V diag(w) Vᵀ.
//
// Unlike eig_op, this specialised routine exploits symmetry for efficiency
// and guarantees real outputs.
//
// Forward dispatch goes to IBackend::linalg_eigh(), which uses LAPACK's
// ssyev/dsyev on the CPU path (single/double precision respectively) and
// mlx::core::linalg::eigh on the GPU path.
// Returns {w, V} where:
//   w has shape (..., n)    — eigenvalues in ascending order
//   V has shape (..., n, n) — columns are the orthonormal eigenvectors
//
// Note: no backward node is registered.  A future backward would implement
// the eigenvector gradient while avoiding sign ambiguity (eigenvectors are
// only defined up to a sign flip).

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute eigenvalues and eigenvectors of a real symmetric square matrix.
//
// Returns {w, V} where w has shape (..., n) and V has shape (..., n, n).
// Eigenvalues in w are in ascending order.  Validates that a is at least 2-D,
// square, and float-typed before dispatching.
LUCID_API std::vector<TensorImplPtr> eigh_op(const TensorImplPtr& a);

}  // namespace lucid
