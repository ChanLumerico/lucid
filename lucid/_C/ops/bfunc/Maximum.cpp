// lucid/_C/ops/bfunc/Maximum.cpp
//
// Implements MaximumBackward::grad_formula and the maximum_op free function.

#include "Maximum.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MaximumBackward::schema_v1{"maximum", 1, AmpPolicy::Promote, true};

// Gradient of element-wise maximum.
//
// The backward pass routes grad_out through a binary gate:
//   mask_a[i] = 1 if a[i] >= b[i], else 0   (a won or tied)
//   mask_b[i] = 1 if a[i] <  b[i], else 0   (b strictly won)
//
// mask_a and mask_b are complementary (they partition the index set), so their
// element-wise sum is always 1.  This avoids double-counting at ties: the
// gradient flows only to a when a[i] == b[i].
//
// saved_inputs_[0] and saved_inputs_[1] are at their original (pre-broadcast)
// shapes here.  ge_mask_storage and lt_mask_storage operate element-wise at
// the size given by n = numel(out_shape_), so the caller is responsible for
// ensuring the shapes match; the BinaryKernel::apply wrapper handles
// reduction to input shapes.
std::pair<Storage, Storage> MaximumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask_a = ge_mask_storage(saved_inputs_[0], saved_inputs_[1], n, dtype_, device_);
    Storage mask_b = lt_mask_storage(saved_inputs_[0], saved_inputs_[1], n, dtype_, device_);
    Storage dx = multiply_storages(grad_out, mask_a, n, dtype_, device_);
    Storage dy = multiply_storages(grad_out, mask_b, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MaximumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MaximumBackward)

}  // namespace lucid
