// lucid/_C/ops/bfunc/Minimum.cpp
//
// Implements MinimumBackward::grad_formula and the minimum_op free function.

#include "Minimum.h"

#include <mlx/ops.h>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "_Extremum.h"

namespace lucid {

const OpSchema MinimumBackward::schema_v1{"minimum", 1, AmpPolicy::Promote, true};

// The reference's split (see _Extremum.h): the smaller operand takes the
// gradient, a tie halves it, a NaN on either side gives it to both.  ``a``
// loses minimum where ``b < a``, so the pair goes into the shared helper
// as ``(b, a)``; the gradients still come back as ``{da, db}``.
//
// Both operands are compared at the output's shape.  Read at their own
// shapes with n = numel(out), an operand broadcast from one element —
// ``clamp`` against a 0-d bound, ``minimum(x, bins[-1])`` — was read past
// its buffer on the CPU.
std::pair<Storage, Storage> MinimumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    const Storage a_b = saved_input_broadcasted(0);
    const Storage b_b = saved_input_broadcasted(1);
    return bfunc_detail::extremum_grads(grad_out, b_b, a_b, n, dtype_, device_);
}

std::pair<TensorImplPtr, TensorImplPtr> MinimumBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    return bfunc_detail::extremum_grads_graph(grad_out, b, a);
}

TensorImplPtr minimum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MinimumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MinimumBackward)

}  // namespace lucid
