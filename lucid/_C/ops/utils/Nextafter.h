// lucid/_C/ops/utils/Nextafter.h
//
// IEEE-754 "next representable float" operation.  ``nextafter(a, b)`` returns
// the float adjacent to ``a`` in the direction of ``b``.  Computation is
// performed on CPU via ``std::nextafter`` regardless of the input device
// because (1) MLX has no equivalent kernel and (2) the bit-level fiddling is
// trivial on a per-element loop.  GPU inputs are round-tripped through CPU.
//
// Both operands must have the same floating-point dtype (F32 or F64).

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr nextafter_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
