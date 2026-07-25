// lucid/_C/ops/ufunc/Arith.cpp
//
// Implementations of the six basic arithmetic unary backward nodes: neg, abs,
// sign, reciprocal, square, cube.  Each section defines the static OpSchema,
// implements grad_formula, provides the public entry-point wrapper, and
// registers the op in the global OpRegistry via LUCID_REGISTER_OP.

#include "Arith.h"

#include <limits>

#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../bfunc/Add.h"
#include "../bfunc/Div.h"
#include "../bfunc/Maximum.h"
#include "../bfunc/Mul.h"
#include "../complex/Imag.h"
#include "../complex/Real.h"
#include "Exponential.h"
#include "ScalarParam.h"

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
    if (!a)
        ErrorBuilder("abs").fail("null input tensor");
    // Complex magnitude |z| = sqrt(Re(z)² + Im(z)²) is a complex→REAL map, so
    // it cannot go through UnaryKernel: that path tags the output with the
    // *input* dtype, which left the real result labelled C64 — the reader then
    // paired consecutive magnitudes into (re, im) and ran off the end of the
    // buffer.  Compose from the real/imag primitives instead, exactly as the
    // sibling `angle` composite does.  Like real/imag/angle (and unlike abs on
    // a real tensor, which keeps its full UnaryKernel gradient), this path is
    // forward-only — Lucid has no complex autograd.
    //
    // Cost: ~9 element-wise kernels instead of one native complex-abs (measured
    // 26 us for 64x1024 on Metal, ~2x the FFT that produced the input).  Going
    // native would need a UnaryKernel hook for a complex->real output dtype
    // *and* a hand-written C64 branch in the CPU backend; reusing ops that are
    // already device-parity tested is the better trade until a profile says
    // otherwise.
    if (a->dtype() == Dtype::C64) {
        auto re = real_op(a);
        auto im = imag_op(a);
        // Scale by m = max(|Re|, |Im|) before squaring: the naive
        // sqrt(Re² + Im²) overflows to inf once |z| > ~1.8e19 and underflows to
        // 0 below ~1e-19, even though both results are representable in F32.
        // |z| = m · sqrt((Re/m)² + (Im/m)²) is exact and overflow-safe.
        auto m = maximum_op(abs_op(re), abs_op(im));
        // Clamping the divisor keeps m == 0 (z == 0) from dividing 0/0; the
        // trailing multiply by the *unclamped* m still yields exactly 0 there.
        // The upper clamp is FLT_MAX, not infinity, so an infinite component
        // divides by a finite scale (inf/inf would be NaN) and propagates inf —
        // matching hypot's contract that |z| is inf if either lane is.  Any
        // finite m is already <= FLT_MAX, so this never perturbs normal inputs.
        const double tiny = static_cast<double>(std::numeric_limits<float>::min());
        const double huge = static_cast<double>(std::numeric_limits<float>::max());
        auto safe_m = clip_op(m, tiny, huge);
        auto r_re = div_op(re, safe_m);
        auto r_im = div_op(im, safe_m);
        return mul_op(m, sqrt_op(add_op(square_op(r_re), square_op(r_im))));
    }
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
