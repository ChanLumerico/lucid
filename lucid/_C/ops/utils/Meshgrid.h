#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API std::vector<TensorImplPtr> meshgrid_op(const std::vector<TensorImplPtr>& xs,
                                                 bool indexing_xy);

}
