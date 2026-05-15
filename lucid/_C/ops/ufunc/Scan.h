// lucid/_C/ops/ufunc/Scan.h
//
// Public entry points for cumulative scan operations along a single axis.
// The backward nodes (CumsumBackward, CumprodBackward) are defined inside
// Scan.cpp and are not exposed through this header.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Inclusive cumulative sum along `axis`.  The output has the same shape as `a`.
// Backward: reverse cumsum — reverse the gradient along axis, apply cumsum,
// then reverse again so that each input position receives the sum of all
// downstream upstream gradients.
LUCID_API TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis);

// Inclusive cumulative product along `axis`.  The output has the same shape as `a`.
// Backward: grad_x_i = sum_j [ grad_y_j * (cumprod_j / x_i) ] which is
// computed via the reverse-cumsum trick on grad_y * cumprod_y, divided by x.
LUCID_API TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis);

// Inclusive cumulative maximum along `axis`.  The output has the same shape as `a`.
// Backward: gradient flows only to the element that first achieved the running max
// at each position (argmax of the prefix up to that point), zero elsewhere.
LUCID_API TensorImplPtr cummax_op(const TensorImplPtr& a, int axis);

// Inclusive cumulative minimum along `axis`.  The output has the same shape as `a`.
// Backward: gradient flows only to the element that first achieved the running min.
LUCID_API TensorImplPtr cummin_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
