// lucid/_C/ops/composite/Logical.h
//
// Boolean AND / OR / XOR / NOT.  Inputs are interpreted as truthy when
// non-zero; outputs are bool tensors.  Composes ``not_equal`` (to coerce to
// bool) with the existing ``bitwise_*`` kernels.  Not differentiable.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr logical_and_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr logical_or_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr logical_xor_op(const TensorImplPtr& a, const TensorImplPtr& b);
LUCID_API TensorImplPtr logical_not_op(const TensorImplPtr& a);

}  // namespace lucid
