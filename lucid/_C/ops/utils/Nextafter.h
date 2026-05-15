// lucid/_C/ops/utils/Nextafter.h
//
// IEEE-754 "next representable float" operation.  ``nextafter(a, b)`` returns
// the float adjacent to ``a`` in the direction of ``b``.  Two compute paths:
//
//   * (GPU, F32): pure MLX bit-twiddle pipeline — view to int32, conditional
//     ±1 step, view back.  No CPU round-trip.
//   * (CPU, F32 or F64) and (GPU, F64): per-element ``std::nextafter`` loop
//     on host memory.  F64 has no GPU path because MLX does not support F64.
//
// Both operands must have the same floating-point dtype (F32 or F64).

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr nextafter_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
