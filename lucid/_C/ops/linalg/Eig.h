#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API std::vector<TensorImplPtr> eig_op(const TensorImplPtr& a);

}
