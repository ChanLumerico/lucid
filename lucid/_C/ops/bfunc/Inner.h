// lucid/_C/ops/bfunc/Inner.h
//
// Declares inner_op, the entry point for the generalised inner product.
// The inner product contracts the last axis of A against the last axis of B:
//   out[i₀,…,iₙ₋₂, j₀,…,jₘ₋₂] = Σ_k A[i₀,…,iₙ₋₂, k] * B[j₀,…,jₘ₋₂, k]
// This matches the semantics of numpy.inner.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the generalised inner product of A and B by contracting their last
// axes.  When gradient tracking is active the computation is lowered to
// einsum_op so that autograd is inherited from the einsum backward pass.
LUCID_API TensorImplPtr inner_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
