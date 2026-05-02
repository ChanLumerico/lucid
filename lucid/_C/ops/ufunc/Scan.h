#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis);

LUCID_API TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
