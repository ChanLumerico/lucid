// lucid/_C/ops/bfunc/Sub.cpp
//
// Implements SubBackward::grad_formula and the sub_op free function.

#include "Sub.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../ops/ufunc/Arith.h"

namespace lucid {

const OpSchema SubBackward::schema_v1{"sub", 1, AmpPolicy::Promote, true};

// Gradient of element-wise subtraction.
//
// dA = grad_out   (identity: clone preserves the gradient tensor for a)
// dB = -grad_out  (negate: increasing b decreases c, so the gradient is
//                  propagated with a sign flip)
//
// Both results are at the broadcast output shape; BinaryKernel::apply reduces
// them to the original input shapes before routing to the leaf accumulators.
std::pair<Storage, Storage> SubBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    return {
        clone_storage(grad_out, n, dtype_, device_),
        negate_storage(grad_out, n, dtype_, device_),
    };
}

std::pair<TensorImplPtr, TensorImplPtr> SubBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& /*a*/, const TensorImplPtr& /*b*/) {
    return {grad_out, neg_op(grad_out)};
}

TensorImplPtr sub_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return SubBackward::forward(a, b);
}

LUCID_REGISTER_OP(SubBackward)

}  // namespace lucid
