// lucid/_C/ops/ufunc/Astype.h
// Element-wise dtype cast: same shape, different element type.
#pragma once
#include "../../api.h"
#include "../../core/Dtype.h"
#include "../../core/fwd.h"
namespace lucid {
LUCID_API TensorImplPtr astype_op(const TensorImplPtr& a, Dtype dst_dtype);
}  // namespace lucid
