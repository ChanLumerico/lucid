// lucid/_C/ops/bfunc/Floordiv.h
//
// Declares floordiv_op, the entry point for element-wise integer floor
// division.  The operation requires equal-shape operands and always returns an
// I64 tensor.  No gradient is defined because floor division is not
// differentiable.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise floor division: out[i] = floor(a[i] / b[i]).
//
// Both inputs must have the same shape and dtype.  The result dtype is always
// I64 regardless of the input dtype.
LUCID_API TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b);

}
