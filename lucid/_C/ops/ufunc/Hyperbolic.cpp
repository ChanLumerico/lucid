#include "Hyperbolic.h"

#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema SinhBackward::schema_v1{"sinh", 1, AmpPolicy::Promote, true};

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

Storage TanhBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage z_sq = square_storage(saved_output_, n, dtype_, device_);
    Storage neg_zsq = mul_scalar_storage(z_sq, -1.0, n, dtype_, device_);
    Storage one_minus = add_scalar_storage(neg_zsq, 1.0, n, dtype_, device_);
    return multiply_storages(g, one_minus, n, dtype_, device_);
}

TensorImplPtr tanh_op(const TensorImplPtr& a) {
    return TanhBackward::forward(a);
}
LUCID_REGISTER_OP(TanhBackward)

}  // namespace lucid
