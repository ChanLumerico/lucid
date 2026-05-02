#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr pad_op(const TensorImplPtr& a,
                               std::vector<std::pair<std::int64_t, std::int64_t>> pad_width,
                               double constant);

}
