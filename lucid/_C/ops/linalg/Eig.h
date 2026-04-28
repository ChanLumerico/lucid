#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Eigendecomposition. Returns [eigenvalues, eigenvectors].
LUCID_API std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a);

}  // namespace lucid
