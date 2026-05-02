#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr tensordot_op(const TensorImplPtr& a,
                                     const TensorImplPtr& b,
                                     std::vector<int> axes_a,
                                     std::vector<int> axes_b);

}
