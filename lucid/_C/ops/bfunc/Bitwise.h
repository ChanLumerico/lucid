// lucid/_C/ops/bfunc/Bitwise.h
//
// Declares the three element-wise bitwise binary operators: AND, OR, XOR.
// These operations require equal-shape operands with integer or Bool dtype.
// No gradient is defined because bitwise operations are not differentiable.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise bitwise AND: out[i] = a[i] & b[i].
LUCID_API TensorImplPtr bitwise_and_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise bitwise OR: out[i] = a[i] | b[i].
LUCID_API TensorImplPtr bitwise_or_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise bitwise XOR: out[i] = a[i] ^ b[i].
LUCID_API TensorImplPtr bitwise_xor_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise left shift: out[i] = a[i] << b[i].
// Bool dtype is rejected — the operation is only defined for integers.
LUCID_API TensorImplPtr bitwise_left_shift_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise right shift: out[i] = a[i] >> b[i].
LUCID_API TensorImplPtr bitwise_right_shift_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
