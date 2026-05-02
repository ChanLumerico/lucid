#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr bitwise_and_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr bitwise_or_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr bitwise_xor_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
