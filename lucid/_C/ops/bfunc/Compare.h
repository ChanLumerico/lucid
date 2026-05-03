// lucid/_C/ops/bfunc/Compare.h
//
// Declares the six element-wise comparison operators: ==, !=, >, >=, <, <=.
// All comparisons require equal-shape operands (no broadcasting) and produce a
// Bool tensor.  None of these operations support autograd because the
// comparison function is not differentiable.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise equality: out[i] = (a[i] == b[i]).
LUCID_API TensorImplPtr equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise inequality: out[i] = (a[i] != b[i]).
LUCID_API TensorImplPtr not_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise strict greater-than: out[i] = (a[i] > b[i]).
LUCID_API TensorImplPtr greater_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise greater-than-or-equal: out[i] = (a[i] >= b[i]).
LUCID_API TensorImplPtr greater_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise strict less-than: out[i] = (a[i] < b[i]).
LUCID_API TensorImplPtr less_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise less-than-or-equal: out[i] = (a[i] <= b[i]).
LUCID_API TensorImplPtr less_equal_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
