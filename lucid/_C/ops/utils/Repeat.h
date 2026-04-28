#pragma once

// =====================================================================
// Repetition: per-element vs whole-array.
//   repeat(x, n, axis) — repeat each element n times along axis
//   tile(x, reps)      — repeat the whole array along each axis
// =====================================================================

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis);
LUCID_API TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps);

}  // namespace lucid
