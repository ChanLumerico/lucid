// lucid/_C/ops/linalg/HouseholderProduct.h
#pragma once
#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {
LUCID_API TensorImplPtr householder_product_op(const TensorImplPtr& H, const TensorImplPtr& tau);
}  // namespace lucid
