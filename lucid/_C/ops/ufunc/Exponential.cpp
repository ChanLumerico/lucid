#include "Exponential.h"

#include "../../core/OpRegistry.h"

namespace lucid {

// --------------- Exp ---------------
const OpSchema ExpBackward::schema_v1{"exp", 1, AmpPolicy::ForceFP32, true};

Storage ExpBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = output * g  (cheap — output is saved, no recompute)
    return multiply_storages(g, saved_output_, n, dtype_, device_);
}

TensorImplPtr exp_op(const TensorImplPtr& a) {
    return ExpBackward::forward(a);
}
LUCID_REGISTER_OP(ExpBackward)

// --------------- Log ---------------
const OpSchema LogBackward::schema_v1{"log", 1, AmpPolicy::ForceFP32, true};

Storage LogBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    return divide_storages(g, saved_inputs_[0], n, dtype_, device_);
}

TensorImplPtr log_op(const TensorImplPtr& a) {
    return LogBackward::forward(a);
}
LUCID_REGISTER_OP(LogBackward)

// --------------- Log2 ---------------
const OpSchema Log2Backward::schema_v1{"log2", 1, AmpPolicy::ForceFP32, true};

Storage Log2Backward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = g / (x * ln2)
    constexpr double kLn2 = 0.69314718055994530941723212145817656807550013436;
    Storage x_ln2 = mul_scalar_storage(saved_inputs_[0], kLn2, n, dtype_, device_);
    return divide_storages(g, x_ln2, n, dtype_, device_);
}

TensorImplPtr log2_op(const TensorImplPtr& a) {
    return Log2Backward::forward(a);
}
LUCID_REGISTER_OP(Log2Backward)

// --------------- Sqrt ---------------
const OpSchema SqrtBackward::schema_v1{"sqrt", 1, AmpPolicy::Promote, true};

Storage SqrtBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = 0.5 * g / sqrt(x) = 0.5 * g / output
    Storage half_g = mul_scalar_storage(g, 0.5, n, dtype_, device_);
    return divide_storages(half_g, saved_output_, n, dtype_, device_);
}

TensorImplPtr sqrt_op(const TensorImplPtr& a) {
    return SqrtBackward::forward(a);
}
LUCID_REGISTER_OP(SqrtBackward)

}  // namespace lucid
