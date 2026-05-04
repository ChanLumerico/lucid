// lucid/_C/ops/bfunc/Div.cpp
//
// Implements DivBackward::grad_formula and the div_op free function.

#include "Div.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../ops/ufunc/Arith.h"
#include "Mul.h"

namespace lucid {

const OpSchema DivBackward::schema_v1{"div", 1, AmpPolicy::Promote, true};

// Gradient of element-wise division (quotient rule).
//
// Given c = a / b:
//   dA = grad_out / b
//   dB = -(grad_out * a) / b²
//
// The computation is broken into explicit storage operations so that each
// intermediate value can be freed individually.  Both saved inputs are first
// expanded to out_shape_ (via saved_input_broadcasted) so that all element-wise
// operations are shape-compatible.
std::pair<Storage, Storage> DivBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);

    auto a_b = saved_input_broadcasted(0);
    auto b_b = saved_input_broadcasted(1);

    // dA = grad_out / b
    Storage dx = divide_storages(grad_out, b_b, n, dtype_, device_);

    // dB = -(grad_out * a) / b²
    Storage b_sq = square_storage(b_b, n, dtype_, device_);
    Storage g_times_a = multiply_storages(grad_out, a_b, n, dtype_, device_);
    Storage div_by_b_sq = divide_storages(g_times_a, b_sq, n, dtype_, device_);
    Storage dy = negate_storage(div_by_b_sq, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

std::pair<TensorImplPtr, TensorImplPtr> DivBackward::grad_formula_impl(
    const TensorImplPtr& grad_out, const TensorImplPtr& a, const TensorImplPtr& b) {
    // da = grad_out / b
    auto da = div_op(grad_out, b);
    // db = -grad_out * a / b^2
    auto b_sq = mul_op(b, b);
    auto ga = mul_op(grad_out, a);
    auto db = neg_op(div_op(ga, b_sq));
    return {da, db};
}

TensorImplPtr div_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return DivBackward::forward(a, b);
}

LUCID_REGISTER_OP(DivBackward)

}  // namespace lucid
