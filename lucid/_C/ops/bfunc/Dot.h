// lucid/_C/ops/bfunc/Dot.h
//
// Declares dot_op, the entry point for the 1-D and 2-D dot-product operation.
// The implementation handles two strictly separate cases:
//   - 1-D × 1-D: inner product yielding a scalar.
//   - 2-D × 2-D: standard matrix multiply equivalent to matmul for rank-2
//     tensors.
// Higher-rank inputs are rejected at runtime.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the dot product of two 1-D vectors or the matrix product of two 2-D
// matrices, with autograd support for both cases.
LUCID_API TensorImplPtr dot_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
