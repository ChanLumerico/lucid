#pragma once

// =====================================================================
// Element-wise comparison ops (output dtype = Bool).
// Mirrors the comparison classes in `lucid/_func/bfunc.py`.
// Forward only — no autograd hookup.
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr equal_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr not_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr greater_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr greater_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr less_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr less_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
