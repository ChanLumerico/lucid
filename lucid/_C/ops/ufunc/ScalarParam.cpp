#include "ScalarParam.h"

#include <cmath>
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"

namespace lucid {

// =================== PowScalar : x ^ exp ===================

const OpSchema PowScalarBackward::schema_v1{"pow_scalar", 1, AmpPolicy::ForceFP32, true};

Storage PowScalarBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage x_pow_em1 = backend::Dispatcher::for_device(device_).pow_scalar(
        saved_inputs_[0], out_shape_, dtype_, exp_ - 1.0);
    Storage scaled = mul_scalar_storage(x_pow_em1, exp_, n, dtype_, device_);
    return multiply_storages(scaled, g, n, dtype_, device_);
}

TensorImplPtr PowScalarBackward::forward(const TensorImplPtr& a, double exp) {
    Validator::input(a, "pow_scalar.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage = backend::Dispatcher::for_device(a->device()).pow_scalar(
        a->storage(), a->shape(), a->dtype(), exp);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()) * 11);

    auto bwd = std::make_shared<PowScalarBackward>();
    bwd->exp_ = exp;
    kernel::NaryKernel<PowScalarBackward, 1>::wire_autograd(std::move(bwd), {a}, out);
    return out;
}

TensorImplPtr pow_scalar_op(const TensorImplPtr& a, double exp) {
    return PowScalarBackward::forward(a, exp);
}
LUCID_REGISTER_OP(PowScalarBackward)

// =================== RPowScalar : base ^ x ===================

const OpSchema RPowScalarBackward::schema_v1{"rpow_scalar", 1, AmpPolicy::ForceFP32, true};

Storage RPowScalarBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    // dx = ln(base) * base^x * g = ln(base) * output * g
    const double ln_base = std::log(base_);
    Storage scaled_out = mul_scalar_storage(saved_output_, ln_base, n, dtype_, device_);
    return multiply_storages(scaled_out, g, n, dtype_, device_);
}

TensorImplPtr RPowScalarBackward::forward(double base, const TensorImplPtr& a) {
    Validator::input(a, "rpow_scalar.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage = backend::Dispatcher::for_device(a->device()).rpow_scalar(
        a->storage(), a->shape(), a->dtype(), base);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()) * 11);

    auto bwd = std::make_shared<RPowScalarBackward>();
    bwd->saved_output_ = out->storage();
    bwd->base_ = base;
    kernel::NaryKernel<RPowScalarBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                             /*save_ins=*/false);
    return out;
}

TensorImplPtr rpow_scalar_op(double base, const TensorImplPtr& a) {
    return RPowScalarBackward::forward(base, a);
}
LUCID_REGISTER_OP(RPowScalarBackward)

// =================== Clip : clamp(x, lo, hi) ===================

const OpSchema ClipBackward::schema_v1{"clip", 1, AmpPolicy::KeepInput, true};

Storage ClipBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask = in_range_mask_storage(saved_inputs_[0], min_, max_, n, dtype_, device_);
    return multiply_storages(g, mask, n, dtype_, device_);
}

TensorImplPtr ClipBackward::forward(const TensorImplPtr& a, double min_v, double max_v) {
    Validator::input(a, "clip.a").non_null();

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    Storage out_storage = backend::Dispatcher::for_device(a->device()).clip(
        a->storage(), a->shape(), a->dtype(), min_v, max_v);
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                            a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    auto bwd = std::make_shared<ClipBackward>();
    bwd->min_ = min_v;
    bwd->max_ = max_v;
    kernel::NaryKernel<ClipBackward, 1>::wire_autograd(std::move(bwd), {a}, out);
    return out;
}

TensorImplPtr clip_op(const TensorImplPtr& a, double min_v, double max_v) {
    return ClipBackward::forward(a, min_v, max_v);
}
LUCID_REGISTER_OP(ClipBackward)

}  // namespace lucid
