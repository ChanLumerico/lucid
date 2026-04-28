#pragma once

// =====================================================================
// Sort / search / extreme-index ops:
//   sort, argsort, argmax, argmin, nonzero, unique, topk
// =====================================================================

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr sort_op(const TensorImplPtr& a, int axis);
LUCID_API TensorImplPtr argsort_op(const TensorImplPtr& a, int axis);
LUCID_API TensorImplPtr argmax_op(const TensorImplPtr& a, int axis, bool keepdims);
LUCID_API TensorImplPtr argmin_op(const TensorImplPtr& a, int axis, bool keepdims);
LUCID_API TensorImplPtr nonzero_op(const TensorImplPtr& a);
LUCID_API TensorImplPtr unique_op(const TensorImplPtr& a);
// Returns {values, indices} — consistent with svd/qr/eig multi-output convention.
LUCID_API std::vector<TensorImplPtr> topk_op(const TensorImplPtr& a, std::int64_t k, int axis);

}  // namespace lucid
