// lucid/_C/ops/utils/Tri.h
//
// Declares the lower-triangular (tril) and upper-triangular (triu) masking ops.
// Both ops zero out elements on the wrong side of a diagonal offset and are
// differentiable: the backward pass applies the same mask to the incoming
// gradient, preserving only the gradient contributions that correspond to
// elements that were retained in the forward pass.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Zero all elements above the k-th diagonal (k=0 is the main diagonal,
// positive k shifts toward the upper-right, negative k toward the lower-left).
// The input must be at least 2-D; the mask is applied to the last two
// dimensions.  Backward: apply tril with the same k to the incoming gradient.
LUCID_API TensorImplPtr tril_op(const TensorImplPtr& a, int k);

// Zero all elements below the k-th diagonal.  Same k convention and
// dimensionality requirements as tril_op.  Backward: apply triu with the same
// k to the incoming gradient.
LUCID_API TensorImplPtr triu_op(const TensorImplPtr& a, int k);

}  // namespace lucid
