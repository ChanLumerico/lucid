#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis);

LUCID_API TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps);

}  // namespace lucid
