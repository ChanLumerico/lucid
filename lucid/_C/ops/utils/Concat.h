#pragma once

// =====================================================================
// Concatenation, stacking, and splitting along an axis.
//   concatenate / stack / hstack / vstack
//   split (n equal pieces) / split_at (at indices) / chunk / unbind
// All forward-only — backward deferred (sum-back grads are simple).
// =====================================================================

#include <cstdint>
#include <utility>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr concatenate_op(const std::vector<TensorImplPtr>& xs,
                                       int axis);
LUCID_API TensorImplPtr stack_op(const std::vector<TensorImplPtr>& xs, int axis);
LUCID_API TensorImplPtr hstack_op(const std::vector<TensorImplPtr>& xs);
LUCID_API TensorImplPtr vstack_op(const std::vector<TensorImplPtr>& xs);

LUCID_API std::vector<TensorImplPtr>
split_op(const TensorImplPtr& a, std::int64_t num_splits, int axis);
LUCID_API std::vector<TensorImplPtr>
split_at_op(const TensorImplPtr& a, std::vector<std::int64_t> indices,
            int axis);
LUCID_API std::vector<TensorImplPtr>
chunk_op(const TensorImplPtr& a, std::int64_t chunks, int axis);
LUCID_API std::vector<TensorImplPtr>
unbind_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
