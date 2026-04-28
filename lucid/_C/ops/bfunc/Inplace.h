#pragma once

// =====================================================================
// In-place binary ops.
//
// Each op computes the same output as its non-in-place counterpart, then
// overwrites `a`'s storage with the result and bumps `a->version()`. The
// returned TensorImplPtr is the same as the input `a`. No autograd hookup —
// in-place ops are forward-only by convention (mirrors `Tensor.add_()` in
// PyTorch, where in-place mutation invalidates the saved tensor for grad).
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr add_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr sub_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr mul_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr div_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr pow_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr maximum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr minimum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
