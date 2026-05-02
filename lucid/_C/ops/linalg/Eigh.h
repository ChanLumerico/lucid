#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Symmetric/Hermitian eigendecomposition.
/// Input must be a real symmetric (or complex Hermitian) square matrix.
/// Returns [eigenvalues (real, ascending), eigenvectors].
/// CPU: LAPACK ssyev / dsyev.  GPU: mlx::core::linalg::eigh.
LUCID_API std::vector<TensorImplPtr> eigh_op(const TensorImplPtr& a);

}  // namespace lucid
