#pragma once

// =====================================================================
// Selection / gathering ops:
//   where(cond, x, y)
//   masked_fill(x, mask, value)
//   roll(x, shifts, axes)
//   gather(x, indices, axis)  — take_along_axis
//   diagonal(x, offset, axis1, axis2)
// =====================================================================

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Where.
LUCID_API TensorImplPtr where_op(const TensorImplPtr& cond,
                                 const TensorImplPtr& x,
                                 const TensorImplPtr& y);
/// Masked fill.
LUCID_API TensorImplPtr masked_fill_op(const TensorImplPtr& a,
                                       const TensorImplPtr& mask,
                                       double value);
/// Roll.
LUCID_API TensorImplPtr roll_op(const TensorImplPtr& a,
                                std::vector<std::int64_t> shifts,
                                std::vector<int> axes);
/// Gather.
LUCID_API TensorImplPtr gather_op(const TensorImplPtr& a, const TensorImplPtr& indices, int axis);
/// Diagonal.
LUCID_API TensorImplPtr diagonal_op(const TensorImplPtr& a, int offset, int axis1, int axis2);

}  // namespace lucid
