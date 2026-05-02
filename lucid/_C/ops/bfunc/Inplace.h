#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr add_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr sub_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr mul_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr div_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr pow_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr maximum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

LUCID_API TensorImplPtr minimum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
