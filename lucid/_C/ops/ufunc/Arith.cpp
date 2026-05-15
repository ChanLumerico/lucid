// lucid/_C/ops/ufunc/Arith.cpp
//
// Implementations of the six basic arithmetic unary backward nodes: neg, abs,
// sign, reciprocal, square, cube.  Each section defines the static OpSchema,
// implements grad_formula, provides the public entry-point wrapper, and
// registers the op in the global OpRegistry via LUCID_REGISTER_OP.

#include "Arith.h"

#include "../../core/OpRegistry.h"
#include "../bfunc/Add.h"
#include "../bfunc/Div.h"
#include "../bfunc/Mul.h"

namespace lucid {

// neg — AmpPolicy::Promote promotes integer inputs to float before dispatch.
const OpSchema NegBackward::schema_v1{"neg", 1, AmpPolicy::Promote, true};

// dL/dx = -dL/dy: negate the upstream gradient in-place over the output shape.
Storage NegBackward::grad_formula(const Storage& g) {
    return negate_storage(g, shape_numel(out_shape_), dtype_, device_);
}

TensorImplPtr
NegBackward::grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr&) {
    return neg_op(g);
}

TensorImplPtr neg_op(const TensorImplPtr& a) {
    return NegBackward::forward(a);
}
LUCID_REGISTER_OP(NegBackward)

// abs — saves input so grad_formula can recompute sign(x).
const OpSchema AbsBackward::schema_v1{"abs", 1, AmpPolicy::Promote, true};

// dL/dx = sign(x) * dL/dy.
Storage AbsBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);
    Storage s = sign_storage(saved_inputs_[0], n, dtype_, device_);
    return multiply_storages(g, s, n, dtype_, device_);
}

TensorImplPtr abs_op(const TensorImplPtr& a) {
    return AbsBackward::forward(a);
}
LUCID_REGISTER_OP(AbsBackward)

// sign — AmpPolicy::KeepInput preserves the input dtype (sign is valid on
// integers).  kHasGradient = false means UnaryKernel never wires autograd, so
// grad_formula is only called if someone constructs a backward node manually;
// the returned empty CpuStorage acts as a zero-gradient sentinel.
const OpSchema SignBackward::schema_v1{"sign", 1, AmpPolicy::KeepInput, true};

// Gradient of sign is zero almost everywhere (discontinuous at 0).
Storage SignBackward::grad_formula(const Storage& g) {
    (void)g;
    return Storage{CpuStorage{}};
}

TensorImplPtr sign_op(const TensorImplPtr& a) {
    return SignBackward::forward(a);
}
LUCID_REGISTER_OP(SignBackward)

// reciprocal — saves input to compute x^2 in the backward pass.
const OpSchema ReciprocalBackward::schema_v1{"reciprocal", 1, AmpPolicy::Promote, true};

// dL/dx = -dL/dy / x^2.
Storage ReciprocalBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage g_div = divide_storages(g, x_sq, n, dtype_, device_);
    return negate_storage(g_div, n, dtype_, device_);
}

TensorImplPtr ReciprocalBackward::grad_formula_impl(const TensorImplPtr& g,
                                                    const TensorImplPtr& x,
                                                    const TensorImplPtr&) {
    // dx = -g / x^2
    auto x_sq = mul_op(x, x);
    return neg_op(div_op(g, x_sq));
}

TensorImplPtr reciprocal_op(const TensorImplPtr& a) {
    return ReciprocalBackward::forward(a);
}
LUCID_REGISTER_OP(ReciprocalBackward)

// square — saves input to compute 2*x in the backward pass.
const OpSchema SquareBackward::schema_v1{"square", 1, AmpPolicy::Promote, true};

// dL/dx = 2*x * dL/dy.
Storage SquareBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage two_x = mul_scalar_storage(saved_inputs_[0], 2.0, n, dtype_, device_);
    return multiply_storages(two_x, g, n, dtype_, device_);
}

TensorImplPtr SquareBackward::grad_formula_impl(const TensorImplPtr& g,
                                                const TensorImplPtr& x,
                                                const TensorImplPtr&) {
    // dx = 2*x * g = (x+x) * g
    return mul_op(add_op(x, x), g);
}

TensorImplPtr square_op(const TensorImplPtr& a) {
    return SquareBackward::forward(a);
}
LUCID_REGISTER_OP(SquareBackward)

// cube — saves input to compute 3*x^2 in the backward pass.
const OpSchema CubeBackward::schema_v1{"cube", 1, AmpPolicy::Promote, true};

// dL/dx = 3*x^2 * dL/dy.
Storage CubeBackward::grad_formula(const Storage& g) {
    const std::size_t n = shape_numel(out_shape_);

    Storage x_sq = square_storage(saved_inputs_[0], n, dtype_, device_);
    Storage three_xsq = mul_scalar_storage(x_sq, 3.0, n, dtype_, device_);
    return multiply_storages(three_xsq, g, n, dtype_, device_);
}

TensorImplPtr cube_op(const TensorImplPtr& a) {
    return CubeBackward::forward(a);
}
LUCID_REGISTER_OP(CubeBackward)

}  // namespace lucid
