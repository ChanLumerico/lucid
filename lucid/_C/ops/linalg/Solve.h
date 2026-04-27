#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
