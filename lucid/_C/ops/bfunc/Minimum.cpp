// lucid/_C/ops/bfunc/Minimum.cpp
//
// Implements MinimumBackward::grad_formula and the minimum_op free function.

#include "Minimum.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MinimumBackward::schema_v1{"minimum", 1, AmpPolicy::Promote, true};

// Gradient of element-wise minimum.
//
// The mask arguments are intentionally swapped relative to MaximumBackward:
// to detect which input is the minimum we test (b >= a) and (b < a) rather
// than (a >= b) and (a < b).
//
//   mask_a[i] = ge_mask(b, a)[i] = (b[i] >= a[i]) ? 1 : 0
//               → 1 when a is the minimum (or tied), 0 otherwise
//   mask_b[i] = lt_mask(b, a)[i] = (b[i] <  a[i]) ? 1 : 0
//               → 1 when b strictly wins, 0 otherwise
//
// This correctly partitions the gradient: exactly one mask element is 1 at
// every position, so the sum of gradients reaching both inputs equals
// grad_out.
std::pair<Storage, Storage> MinimumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);

    // Arguments to ge/lt are (b, a), not (a, b) — this is the key difference
    // from MaximumBackward which uses (a, b).
    Storage mask_a = ge_mask_storage(saved_inputs_[1], saved_inputs_[0], n, dtype_, device_);
    Storage mask_b = lt_mask_storage(saved_inputs_[1], saved_inputs_[0], n, dtype_, device_);
    Storage dx = multiply_storages(grad_out, mask_a, n, dtype_, device_);
    Storage dy = multiply_storages(grad_out, mask_b, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

TensorImplPtr minimum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MinimumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MinimumBackward)

}  // namespace lucid
