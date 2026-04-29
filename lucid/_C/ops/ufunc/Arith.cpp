#include "Arith.h"

#include "../../core/OpRegistry.h"

namespace lucid {

// --------------- Neg ---------------
const OpSchema NegBackward::schema_v1{"neg", 1, AmpPolicy::Promote, true};

Storage NegBackward::grad_formula(const Storage& g) {
    return negate_storage(g, shape_numel(out_shape_), dtype_, device_);
}

TensorImplPtr neg_op(const TensorImplPtr& a) {
    return NegBackward::forward(a);
}
LUCID_REGISTER_OP(NegBackward)

// --------------- Abs ---------------
const OpSchema AbsBackward::schema_v1{"abs", 1, AmpPolicy::Promote, true};

Storage AbsBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage s = sign_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, s, n, dtype_, device_);
}

TensorImplPtr abs_op(const TensorImplPtr& a) {
    return AbsBackward::forward(a);
}
LUCID_REGISTER_OP(AbsBackward)

// --------------- Sign (no grad) ---------------
const OpSchema SignBackward::schema_v1{"sign", 1, AmpPolicy::KeepInput, true};

Storage SignBackward::grad_formula(const Storage& g) {
    // Never called — kHasGradient = false.
    (void)g;
    return Storage{CpuStorage{}};
}

TensorImplPtr sign_op(const TensorImplPtr& a) {
    return SignBackward::forward(a);
}
LUCID_REGISTER_OP(SignBackward)

// --------------- Reciprocal ---------------
const OpSchema ReciprocalBackward::schema_v1{"reciprocal", 1, AmpPolicy::Promote, true};

Storage ReciprocalBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = -1/x² * g
    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage g_div = divide_storages(g, x_sq, n, dtype_, device_);
    return negate_storage(g_div, n, dtype_, device_);
}

TensorImplPtr reciprocal_op(const TensorImplPtr& a) {
    return ReciprocalBackward::forward(a);
}
LUCID_REGISTER_OP(ReciprocalBackward)

// --------------- Square ---------------
const OpSchema SquareBackward::schema_v1{"square", 1, AmpPolicy::Promote, true};

Storage SquareBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = 2 * x * g
    Storage two_x = mul_scalar_storage(saved_inputs_[0], 2.0, n, dtype_, device_);
    return multiply_storages(two_x, g, n, dtype_, device_);
}

TensorImplPtr square_op(const TensorImplPtr& a) {
    return SquareBackward::forward(a);
}
LUCID_REGISTER_OP(SquareBackward)

// --------------- Cube ---------------
const OpSchema CubeBackward::schema_v1{"cube", 1, AmpPolicy::Promote, true};

Storage CubeBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = 3 * x^2 * g
    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage three_xsq = mul_scalar_storage(x_sq, 3.0, n, dtype_, device_);
    return multiply_storages(three_xsq, g, n, dtype_, device_);
}

TensorImplPtr cube_op(const TensorImplPtr& a) {
    return CubeBackward::forward(a);
}
LUCID_REGISTER_OP(CubeBackward)

}  // namespace lucid
