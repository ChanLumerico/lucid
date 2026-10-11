// lucid/_C/ops/bfunc/Maximum.cpp
//
// Implements MaximumBackward::grad_formula and the maximum_op free function.

#include "Maximum.h"

#include <mlx/ops.h>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "_Extremum.h"

namespace lucid {

const OpSchema MaximumBackward::schema_v1{"maximum", 1, AmpPolicy::Promote, true};

// The reference's split (see _Extremum.h): the larger operand takes the
// gradient, a tie halves it, a NaN on either side gives it to both.
//
// Both operands are compared at the output's shape.  Read at their own
// shapes with n = numel(out), an operand broadcast from one element —
// ``clamp`` against a 0-d bound, ``minimum(x, bins[-1])`` — was read past
// its buffer on the CPU.  The BinaryKernel::apply wrapper reduces each
// gradient to its input's shape.
std::pair<Storage, Storage> MaximumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    const Storage a_b = saved_input_broadcasted(0);
    const Storage b_b = saved_input_broadcasted(1);
    return bfunc_detail::extremum_grads(grad_out, a_b, b_b, n, dtype_, device_);
}

// Graph-mode derivative: the same split from tracked ops, so
// ``grad(create_graph=True)`` reaches every composite written on top —
// ``hypot``, ``logaddexp``, ``celu``, ``prelu``.
std::pair<TensorImplPtr, TensorImplPtr> MaximumBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    return bfunc_detail::extremum_grads_graph(grad_out, a, b);
}

TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MaximumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MaximumBackward)

}  // namespace lucid
