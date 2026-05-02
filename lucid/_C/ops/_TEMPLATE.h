

#pragma once

#include <memory>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr my_op(const TensorImplPtr& a, const TensorImplPtr& b);

}
