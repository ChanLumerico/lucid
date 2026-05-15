// lucid/_C/ops/ufunc/Hyperbolic.cpp
//
// Gradient formulas and entry points for sinh, cosh, and tanh.
//
// sinh and cosh both save the input and call the complementary function in
// their backward pass (sinh→cosh_storage, cosh→sinh_storage).
// tanh saves the *output* instead of the input because the formula
// dL/dx = (1 - y^2) * dL/dy only needs y, which avoids a redundant vvtanhf
// call into vForce.  The sech^2 factor (1 - y^2) is reconstructed from
// storage primitives without a dedicated subtract kernel.

#include "Hyperbolic.h"

#include "../../core/OpRegistry.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"

namespace lucid {

const OpSchema SinhBackward::schema_v1{"sinh", 1, AmpPolicy::Promote, true};

// dL/dx = cosh(x) * dL/dy.
Storage SinhBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage cx = cosh_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, cx, n, dtype_, device_);
}

TensorImplPtr sinh_op(const TensorImplPtr& a) {
    return SinhBackward::forward(a);
}
LUCID_REGISTER_OP(SinhBackward)

const OpSchema CoshBackward::schema_v1{"cosh", 1, AmpPolicy::Promote, true};

// dL/dx = sinh(x) * dL/dy.
Storage CoshBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sx = sinh_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, sx, n, dtype_, device_);
}

TensorImplPtr cosh_op(const TensorImplPtr& a) {
    return CoshBackward::forward(a);
}
LUCID_REGISTER_OP(CoshBackward)

const OpSchema TanhBackward::schema_v1{"tanh", 1, AmpPolicy::Promote, true};

// dL/dx = (1 - y^2) * dL/dy, computed as ((-y^2) + 1) * dL/dy.
// Building (1 - y^2) via mul_scalar(-1) + add_scalar(1) avoids a subtraction
// kernel call; the three-step sequence reuses existing storage primitives.
Storage TanhBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage z_sq = square_storage(saved_output_, n, dtype_, device_);
    Storage neg_zsq = mul_scalar_storage(z_sq, -1.0, n, dtype_, device_);
    Storage one_minus = add_scalar_storage(neg_zsq, 1.0, n, dtype_, device_);
    return multiply_storages(g, one_minus, n, dtype_, device_);
}

TensorImplPtr TanhBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr&,
                                              const TensorImplPtr& out) {
    // dx = (1 - out^2) * g
    auto out_sq = mul_op(out, out);
    auto one_minus = sub_op(ones_like_op(out), out_sq);
    return mul_op(g, one_minus);
}

TensorImplPtr tanh_op(const TensorImplPtr& a) {
    return TanhBackward::forward(a);
}
LUCID_REGISTER_OP(TanhBackward)

}  // namespace lucid
