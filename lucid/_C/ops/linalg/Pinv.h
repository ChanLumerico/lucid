#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Pinv.
LUCID_API TensorImplPtr pinv_op(const TensorImplPtr& a);

}  // namespace lucid
