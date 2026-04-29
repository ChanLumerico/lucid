#pragma once

// =====================================================================
// Element-wise bitwise ops (input dtype must be integer or bool).
// Mirrors `_bitwise_and`, `_bitwise_or`, `_bitwise_xor` in
// `lucid/_func/bfunc.py`. Forward only — no autograd.
// =====================================================================

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Bitwise and.
LUCID_API TensorImplPtr bitwise_and_op(const TensorImplPtr& a, const TensorImplPtr& b);
/// Bitwise or.
LUCID_API TensorImplPtr bitwise_or_op(const TensorImplPtr& a, const TensorImplPtr& b);
/// Bitwise xor.
LUCID_API TensorImplPtr bitwise_xor_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
