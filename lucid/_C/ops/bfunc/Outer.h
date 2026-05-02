#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr outer_op(const TensorImplPtr& a, const TensorImplPtr& b);

}
