// lucid/_C/ops/ufunc/Trig.cpp
//
// Gradient formulas for the six trigonometric ops.  Each implementation only
// uses cheap storage-level primitives (cos_storage, square_storage, …) — no
// temporary TensorImpl allocations — to keep backward overhead minimal.

#include "Trig.h"

#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema SinBackward::schema_v1{"sin", 1, AmpPolicy::Promote, true};

// dL/dx = cos(x) * dL/dy.
Storage SinBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage cosx = cos_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, cosx, n, dtype_, device_);
}

TensorImplPtr sin_op(const TensorImplPtr& a) {
    return SinBackward::forward(a);
}
LUCID_REGISTER_OP(SinBackward)

const OpSchema CosBackward::schema_v1{"cos", 1, AmpPolicy::Promote, true};

// dL/dx = -sin(x) * dL/dy.
Storage CosBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage sinx = sin_storage(saved_inputs_[0], n, dtype_, device_);
    Storage neg_sin = negate_storage(sinx, n, dtype_, device_);
    return multiply_storages(g, neg_sin, n, dtype_, device_);
}

TensorImplPtr cos_op(const TensorImplPtr& a) {
    return CosBackward::forward(a);
}
LUCID_REGISTER_OP(CosBackward)

const OpSchema TanBackward::schema_v1{"tan", 1, AmpPolicy::Promote, true};

// dL/dx = dL/dy / cos^2(x)  (equivalent to dL/dy * sec^2(x)).
Storage TanBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage cosx = cos_storage(saved_inputs_[0], n, dtype_, device_);
    Storage cos_sq = square_storage(cosx, n, dtype_, device_);
    return divide_storages(g, cos_sq, n, dtype_, device_);
}

TensorImplPtr tan_op(const TensorImplPtr& a) {
    return TanBackward::forward(a);
}
LUCID_REGISTER_OP(TanBackward)

const OpSchema AsinBackward::schema_v1{"arcsin", 1, AmpPolicy::Promote, true};

// dL/dx = dL/dy / sqrt(1 - x^2).
// Building the radicand as (-x^2 + 1) with mul_scalar(-1) + add_scalar(1)
// avoids a separate subtract kernel.
Storage AsinBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage one_min = mul_scalar_storage(x_sq, -1.0, n, dtype_, device_);
    Storage radicand = add_scalar_storage(one_min, 1.0, n, dtype_, device_);
    Storage root = sqrt_storage(radicand, n, dtype_, device_);
    return divide_storages(g, root, n, dtype_, device_);
}

TensorImplPtr arcsin_op(const TensorImplPtr& a) {
    return AsinBackward::forward(a);
}
LUCID_REGISTER_OP(AsinBackward)

const OpSchema AcosBackward::schema_v1{"arccos", 1, AmpPolicy::Promote, true};

// dL/dx = -dL/dy / sqrt(1 - x^2).
// Same radicand construction as arcsin, plus a final negation.
Storage AcosBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage one_min = mul_scalar_storage(x_sq, -1.0, n, dtype_, device_);
    Storage radicand = add_scalar_storage(one_min, 1.0, n, dtype_, device_);
    Storage root = sqrt_storage(radicand, n, dtype_, device_);
    Storage q = divide_storages(g, root, n, dtype_, device_);
    return negate_storage(q, n, dtype_, device_);
}

TensorImplPtr arccos_op(const TensorImplPtr& a) {
    return AcosBackward::forward(a);
}
LUCID_REGISTER_OP(AcosBackward)

const OpSchema AtanBackward::schema_v1{"arctan", 1, AmpPolicy::Promote, true};

// dL/dx = dL/dy / (1 + x^2).
Storage AtanBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage denom = add_scalar_storage(x_sq, 1.0, n, dtype_, device_);
    return divide_storages(g, denom, n, dtype_, device_);
}

TensorImplPtr arctan_op(const TensorImplPtr& a) {
    return AtanBackward::forward(a);
}
LUCID_REGISTER_OP(AtanBackward)

}  // namespace lucid
