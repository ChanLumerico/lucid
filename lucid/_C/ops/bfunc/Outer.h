// lucid/_C/ops/bfunc/Outer.h
//
// Declares outer_op, the entry point for the outer (tensor) product of two
// 1-D vectors.  The outer product produces a 2-D matrix:
//   C[i, j] = a[i] * b[j]

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the outer product of two 1-D vectors a (length M) and b (length N),
// yielding a [M×N] matrix.  Autograd is supported via OuterBackward.
LUCID_API TensorImplPtr outer_op(const TensorImplPtr& a, const TensorImplPtr& b);

}
