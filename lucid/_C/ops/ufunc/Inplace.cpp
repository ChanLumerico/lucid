#include "Inplace.h"

#include <utility>

#include "../../core/Exceptions.h"
#include "../../core/TensorImpl.h"
#include "Activation.h"
#include "Arith.h"
#include "Discrete.h"
#include "Exponential.h"
#include "Hyperbolic.h"
#include "ScalarParam.h"
#include "Trig.h"

namespace lucid {

namespace {

template <typename Fn>
TensorImplPtr inplace_unary(const TensorImplPtr& a, Fn&& fwd_fn,
                            const char* name) {
    if (!a) throw LucidError(std::string(name) + ": null input");
    auto out = fwd_fn(a);
    if (out->shape_ != a->shape_)
        throw ShapeMismatch(a->shape_, out->shape_,
                            std::string(name) + " (in-place: shape changed)");
    a->storage_ = std::move(out->storage_);
    a->dtype_   = out->dtype_;
    a->device_  = out->device_;
    a->version_ += 1;
    return a;
}

}  // namespace

// -- Arith --
TensorImplPtr neg_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &neg_op, "neg_");
}
TensorImplPtr abs_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &abs_op, "abs_");
}
TensorImplPtr sign_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sign_op, "sign_");
}
TensorImplPtr reciprocal_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &reciprocal_op, "reciprocal_");
}
TensorImplPtr square_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &square_op, "square_");
}
TensorImplPtr cube_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &cube_op, "cube_");
}

// -- Exponential / log --
TensorImplPtr exp_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &exp_op, "exp_");
}
TensorImplPtr log_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &log_op, "log_");
}
TensorImplPtr log2_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &log2_op, "log2_");
}
TensorImplPtr sqrt_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sqrt_op, "sqrt_");
}

// -- Trig --
TensorImplPtr sin_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sin_op, "sin_");
}
TensorImplPtr cos_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &cos_op, "cos_");
}
TensorImplPtr tan_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &tan_op, "tan_");
}
TensorImplPtr arcsin_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &arcsin_op, "arcsin_");
}
TensorImplPtr arccos_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &arccos_op, "arccos_");
}
TensorImplPtr arctan_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &arctan_op, "arctan_");
}

// -- Hyperbolic --
TensorImplPtr sinh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sinh_op, "sinh_");
}
TensorImplPtr cosh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &cosh_op, "cosh_");
}
TensorImplPtr tanh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &tanh_op, "tanh_");
}

// -- Discrete --
TensorImplPtr round_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &round_op, "round_");
}
TensorImplPtr floor_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &floor_op, "floor_");
}
TensorImplPtr ceil_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &ceil_op, "ceil_");
}

// -- Scalar-parameterized --
TensorImplPtr clip_inplace_op(const TensorImplPtr& a, double lo, double hi) {
    if (!a) throw LucidError("clip_: null input");
    auto out = clip_op(a, lo, hi);
    a->storage_ = std::move(out->storage_);
    a->dtype_   = out->dtype_;
    a->device_  = out->device_;
    a->version_ += 1;
    return a;
}

}  // namespace lucid
