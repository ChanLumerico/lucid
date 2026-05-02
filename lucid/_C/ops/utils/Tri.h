#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr tril_op(const TensorImplPtr& a, int k);

LUCID_API TensorImplPtr triu_op(const TensorImplPtr& a, int k);

}  // namespace lucid
