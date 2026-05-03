// lucid/_C/ops/utils/Select.h
//
// Declares element-selection and index-based ops: where, masked_fill, roll,
// gather, and diagonal.  Several of these are differentiable; their backward
// passes are implemented in Select.cpp.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Select elements from `x` where `cond` is true, else from `y`.  `cond`,
// `x`, and `y` must share the same device.  On GPU, broadcasting is handled
// by the MLX backend; on CPU, all three must be the same shape.
// Backward: propagate grad_out through the x-branch (cond=true) and
// y-branch (cond=false) independently by masking with cond and ~cond.
LUCID_API TensorImplPtr where_op(const TensorImplPtr& cond,
                                 const TensorImplPtr& x,
                                 const TensorImplPtr& y);

// Fill positions of `a` where `mask` is true with scalar `value`.  The mask
// must have the same shape as `a`.  Backward: propagate grad_out only through
// positions where mask is false (the positions that were not overwritten).
LUCID_API TensorImplPtr masked_fill_op(const TensorImplPtr& a,
                                       const TensorImplPtr& mask,
                                       double value);

// Circularly shift `a` by `shifts[i]` elements along `axes[i]` for each
// entry.  `shifts` and `axes` must have equal length.  Backward: roll with
// negated shifts along the same axes, which exactly inverts a circular shift.
LUCID_API TensorImplPtr roll_op(const TensorImplPtr& a,
                                std::vector<std::int64_t> shifts,
                                std::vector<int> axes);

// Gather values from `a` at positions given by `indices` along `axis`.
// `a` and `indices` must have the same rank.  The output shape equals
// `indices.shape`.  Backward: scatter-add the gradient back to the input
// positions using the same indices (GatherBackward calls
// Dispatcher::gather_backward, which is a scatter_add operation).
LUCID_API TensorImplPtr gather_op(const TensorImplPtr& a, const TensorImplPtr& indices, int axis);

// Extract the diagonal of `a` with the given offset along (axis1, axis2).
// `a` must be at least 2-D.  The two axes are canonicalised so that a1 < a2.
// The output places the diagonal elements in a final appended dimension of
// length L = max(0, min(M - r0, N - c0)) where M, N are the sizes of the two
// selected axes and r0, c0 are the starting row/column for the given offset.
// Backward: scatter the diagonal gradient back into a zero tensor of the
// original input shape.
LUCID_API TensorImplPtr diagonal_op(const TensorImplPtr& a, int offset, int axis1, int axis2);

}  // namespace lucid
