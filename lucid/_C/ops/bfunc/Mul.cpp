// lucid/_C/ops/bfunc/Mul.cpp
//
// Implements MulBackward::grad_formula and the mul_op free function.

#include "Mul.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MulBackward::schema_v1{"mul", 1, AmpPolicy::Promote, true};

// Gradient of element-wise multiplication (product rule).
//
// saved_input_broadcasted(k) returns saved_inputs_[k] already expanded to
// out_shape_ if the original input was narrower due to broadcasting.  This
// expansion is required so that element-wise multiplication with grad_out (also
// at out_shape_) is shape-compatible.
//
// dA = grad_out ⊙ b_broadcast
// dB = grad_out ⊙ a_broadcast
//
// BinaryKernel::apply will subsequently reduce both results back to the
// original input shapes via reduce_grad_to_shape.
std::pair<Storage, Storage> MulBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);

    auto a_b = saved_input_broadcasted(0);
    auto b_b = saved_input_broadcasted(1);
    return {
        multiply_storages(grad_out, b_b, n, dtype_, device_),
        multiply_storages(grad_out, a_b, n, dtype_, device_),
    };
}

std::pair<TensorImplPtr, TensorImplPtr> MulBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    // da = grad_out * b,  db = grad_out * a.
    // a and b are already broadcast-expanded to out_shape_ by BinaryKernel.
    return {mul_op(grad_out, b), mul_op(grad_out, a)};
}

TensorImplPtr mul_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MulBackward::forward(a, b);
}

LUCID_REGISTER_OP(MulBackward)

}  // namespace lucid
