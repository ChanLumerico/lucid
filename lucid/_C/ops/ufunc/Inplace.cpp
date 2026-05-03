// lucid/_C/ops/ufunc/Inplace.cpp
//
// Implementation of all in-place unary ops.
//
// The anonymous-namespace helper inplace_unary<Fn> encapsulates the shared
// write-back pattern:
//   1. Run the corresponding out-of-place op (fwd_fn) to get a fresh tensor.
//   2. Assert that the output shape matches the input shape (in-place ops must
//      not change shape).
//   3. Move the new storage, dtype, and device back into the source tensor `a`.
//   4. Bump `a`'s version counter so autograd can detect illegal mutations.
//
// clip_inplace_op is handled separately because it requires two scalar
// parameters (lo, hi) that cannot be passed through the zero-argument Fn type.

#include "Inplace.h"

#include <utility>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "Activation.h"
#include "Arith.h"
#include "Discrete.h"
#include "Exponential.h"
#include "Hyperbolic.h"
#include "ScalarParam.h"
#include "Trig.h"

namespace lucid {

namespace {

// Run fwd_fn(a), verify the shape did not change, then write the result back
// into `a` and bump its version counter.  Returns `a` (same pointer).
template <typename Fn>
TensorImplPtr inplace_unary(const TensorImplPtr& a, Fn&& fwd_fn, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    auto out = fwd_fn(a);
    if (out->shape() != a->shape())
        throw ShapeMismatch(a->shape(), out->shape(),
                            std::string(name) + " (in-place: shape changed)");
    a->mutable_storage() = std::move(out->mutable_storage());
    a->set_dtype(out->dtype());
    a->set_device(out->device());
    a->bump_version();
    return a;
}

}  // namespace

// --- Arithmetic ---
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

// --- Exponential / Logarithm ---
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

// --- Trigonometric ---
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

// --- Hyperbolic ---
TensorImplPtr sinh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &sinh_op, "sinh_");
}
TensorImplPtr cosh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &cosh_op, "cosh_");
}
TensorImplPtr tanh_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &tanh_op, "tanh_");
}

// --- Rounding ---
TensorImplPtr round_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &round_op, "round_");
}
TensorImplPtr floor_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &floor_op, "floor_");
}
TensorImplPtr ceil_inplace_op(const TensorImplPtr& a) {
    return inplace_unary(a, &ceil_op, "ceil_");
}

// Clip is handled manually because the lambda would need to capture lo/hi,
// making it incompatible with the function-pointer-based inplace_unary template.
TensorImplPtr clip_inplace_op(const TensorImplPtr& a, double lo, double hi) {
    Validator::input(a, "clip_.a").non_null();
    auto out = clip_op(a, lo, hi);
    a->mutable_storage() = std::move(out->mutable_storage());
    a->set_dtype(out->dtype());
    a->set_device(out->device());
    a->bump_version();
    return a;
}

}  // namespace lucid
