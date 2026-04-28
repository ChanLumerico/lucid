#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr norm_op(const TensorImplPtr& a,
                                double ord,
                                std::vector<int> axis,
                                bool keepdims);

}  // namespace lucid
