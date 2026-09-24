// lucid/_C/ops/bfunc/Maximum.cpp
//
// Implements MaximumBackward::grad_formula and the maximum_op free function.

#include "Maximum.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Select.h"
#include "Compare.h"

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
// The masks compare the operands at the output's shape
// (``saved_input_broadcasted``), element-wise over n = numel(out_shape_);
// the BinaryKernel::apply wrapper reduces each gradient to its input's shape.
std::pair<Storage, Storage> MaximumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    // Both operands at the output's shape.  They were read at their own
    // shapes with n = numel(out), so an operand broadcast from one element
    // — ``clamp`` against a 0-d bound, ``minimum(x, bins[-1])`` — was read
    // past its buffer on the CPU and the gradient masked against garbage.
    const Storage a_b = saved_input_broadcasted(0);
    const Storage b_b = saved_input_broadcasted(1);
    Storage mask_a = ge_mask_storage(a_b, b_b, n, dtype_, device_);
    Storage mask_b = lt_mask_storage(a_b, b_b, n, dtype_, device_);
    Storage dx = multiply_storages(grad_out, mask_a, n, dtype_, device_);
    Storage dy = multiply_storages(grad_out, mask_b, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

// Graph-mode derivative.
//
// This had an eager ``grad_formula`` and no ``grad_formula_impl``, so
// ``grad(create_graph=True)`` refused it — and with it every composite
// written on top: ``clamp``, ``clip``, ``hypot``, ``logaddexp``,
// ``celu``, ``prelu``.
//
// The gradient goes entirely to the operand that won.  Ties go to the
// first, matching the forward, so the two branches sum to exactly the
// incoming gradient and nothing is created or lost.
std::pair<TensorImplPtr, TensorImplPtr> MaximumBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    auto a_wins = greater_equal_op(a, b);
    auto zero = zeros_like_op(grad_out);
    return {where_op(a_wins, grad_out, zero), where_op(a_wins, zero, grad_out)};
}

TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MaximumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MaximumBackward)

}  // namespace lucid
