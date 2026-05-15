// lucid/_C/ops/composite/Reductions.h
//
// Reduction operations expressed as compositions of primitive reductions.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Numerically-stable ``log(sum(exp(x)))`` via the max-shift trick:
//
//     m   = max(x, axes, keepdims=true)
//     out = log(sum(exp(x − m), axes, keepdims=true)) + m
//
// The intermediate keeps the reduced axes so the broadcast subtract is well
// defined; if ``keepdims`` is false those axes are squeezed off the final
// output.  Forward-only composition — gradient flows through ``MaxBackward``,
// ``SubBackward``, ``ExpBackward``, ``SumBackward``, ``LogBackward`` and
// ``AddBackward`` automatically.
LUCID_API TensorImplPtr logsumexp_op(const TensorImplPtr& a,
                                     const std::vector<int>& axes,
                                     bool keepdims);

}  // namespace lucid
