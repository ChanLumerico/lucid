#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API std::vector<TensorImplPtr> svd_op(const TensorImplPtr& a, bool compute_uv = true);

}
