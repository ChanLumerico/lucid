#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr var_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

}
