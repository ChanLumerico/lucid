// lucid/_C/ops/composite/Math.cpp
//
// Composition-based elementwise math ops.  Each entry function builds its
// forward graph from existing differentiable ``*_op`` calls; the chain rule
// is provided automatically by the primitives' backward nodes (LogBackward,
// ExpBackward, MulBackward, etc.) so no new schemas are registered here.
//
// Each opens with an :class:`AmpScope` under its own name, so autocast
// applies the composite's row once to its inputs and the primitives inside
// compute in that dtype.

#include "Math.h"

#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/Add.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Div.h"
#include "../bfunc/Maximum.h"
#include "../bfunc/Minimum.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Arith.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Discrete.h"
#include "../ufunc/Exponential.h"
#include "../ufunc/Predicate.h"
#include "../ufunc/Trig.h"
#include "../utils/Promote.h"
#include "../utils/Select.h"

namespace lucid {

namespace {

// The operand of a unary composite, null-checked before its device is read.
const TensorImplPtr& unary_input(const TensorImplPtr& a, const char* op) {
    if (!a)
        ErrorBuilder(op).fail("null input");
    return a;
}

// The two operands of a binary composite, cast together by its scope.
std::pair<TensorImplPtr, TensorImplPtr>
cast_operands(const AmpScope& amp, const TensorImplPtr& a, const TensorImplPtr& b) {
    auto cast = amp.cast_widest({a, b});
    return {std::move(cast[0]), std::move(cast[1])};
}

}  // namespace

// ── unary compositions ──────────────────────────────────────────────────────

// ``log10(x) = log(x) / log(10)``.  Materialises ``log(10)`` as a same-shape
// constant so the divide kernel does not need scalar broadcasting.
TensorImplPtr log10_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"log10", unary_input(a_in, "log10")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    auto log_a = log_op(a);
    auto ln10 = full_like_op(log_a, std::log(10.0));
    return div_op(log_a, ln10);
}

// ``log1p(x) = log(1 + x)`` — but not by forming ``1 + x`` and taking the
// log of that, which is where the information goes.  Below the machine
// epsilon ``1 + x`` rounds to exactly 1 and the answer comes back 0 where
// it should be x: ``log1p(1e-30)`` was 0.0 against the reference's 1e-30,
// and everything under about 1e-8 had lost all of its digits.
//
// The standard repair takes the log at the rounded operand and rescales
// by the increment the rounding actually applied:
//
//     u = 1 + x;   log1p(x) = (u - 1 == 0) ? x : log(u) · x / (u - 1)
//
// ``u - 1`` recovers exactly what the addition put in — it is not
// algebraically ``x``, and that difference is the correction.
//
// The rescaling is a value correction only.  Differentiated, it is
// 0/0 or inf/inf wherever ``log(u)`` is not finite: the value at +inf was
// NaN (inf · inf / inf), and the gradient was NaN at x = -1 and below,
// where the derivative is 1 / (1 + x) — +inf at -1, finite beyond.  So
// the graph carries ``log(u)``, whose backward is exactly 1 / (1 + x),
// and the correction is added outside autograd, and only where
// ``log(u)`` is finite; elsewhere ``log(u)`` already is the answer
// (+inf, -inf at -1, NaN below -1 and at NaN).  ``x / (u - 1)`` is
// divided before it multiplies, since it is about 1 and ``log(u) · x``
// overflows float32 from x ≈ 4e36.
TensorImplPtr log1p_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"log1p", unary_input(a_in, "log1p")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    auto ones = full_like_op(a, 1.0);
    auto u = add_op(a, ones);
    auto plain = log_op(u);
    TensorImplPtr vanished;
    TensorImplPtr correction;
    {
        NoGradGuard no_grad;
        auto increment = sub_op(u, ones);
        vanished = equal_op(increment, full_like_op(increment, 0.0));
        auto safe = where_op(vanished, ones, increment);
        auto accurate = mul_op(plain, div_op(a, safe));
        correction =
            where_op(isfinite_op(plain), sub_op(accurate, plain), full_like_op(plain, 0.0));
    }
    // Where the increment vanished, x is already the answer to within one
    // ulp, its derivative 1 / (1 + x) rounds to 1, and taking x keeps
    // log1p(-0) = -0.
    return where_op(vanished, a, add_op(plain, correction));
}

// ``exp2(x) = exp(x · log(2))``.  Multiplying against a constant tensor keeps
// the gradient chain visible to ``MulBackward`` and ``ExpBackward``.
TensorImplPtr exp2_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"exp2", unary_input(a_in, "exp2")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    auto ln2 = full_like_op(a, std::log(2.0));
    return exp_op(mul_op(a, ln2));
}

// Fractional part — the ``trunc`` term contributes zero gradient, so the
// gradient w.r.t. ``a`` is just the upstream gradient.
TensorImplPtr frac_op(const TensorImplPtr& a_in) {
    const AmpScope amp{"frac", unary_input(a_in, "frac")->device()};
    const TensorImplPtr a = amp.cast(a_in);
    return sub_op(a, trunc_op(a));
}

// ── binary compositions ─────────────────────────────────────────────────────

// Quadrant-aware arctangent.  Patches ``arctan(y / x)`` with ±π/zero/±π/2
// branches via ``where`` based on the signs of the operands; the divide is
// guarded by substituting 1 for x = 0 so the safe branch is always defined.
TensorImplPtr atan2_op(const TensorImplPtr& y_in, const TensorImplPtr& x_in) {
    constexpr double pi = 3.14159265358979323846;
    if (!y_in || !x_in)
        ErrorBuilder("atan2").fail("null input");
    const auto [y_dev, x_dev] = detail::align_binary_devices(y_in, x_in, "atan2");
    const AmpScope amp{"atan2", y_dev->device()};
    const auto [y, x] = cast_operands(amp, y_dev, x_dev);

    auto zero_x = full_like_op(x, 0.0);
    auto zero_y = full_like_op(y, 0.0);
    auto ones_x = full_like_op(x, 1.0);

    auto x_is_zero = equal_op(x, zero_x);
    auto safe_x = where_op(x_is_zero, ones_x, x);
    auto base = arctan_op(div_op(y, safe_x));

    auto x_pos = greater_op(x, zero_x);
    auto y_neg = less_op(y, zero_y);
    auto y_pos = greater_op(y, zero_y);
    auto y_zero = equal_op(y, zero_y);

    auto base_plus_pi = add_op(base, full_like_op(base, pi));
    auto base_minus_pi = sub_op(base, full_like_op(base, pi));
    auto half_pi = full_like_op(base, 0.5 * pi);
    auto neg_half_pi = full_like_op(base, -0.5 * pi);
    auto zero_out = full_like_op(base, 0.0);

    // x > 0 keeps base; x < 0 adds ±π depending on the sign of y.
    auto x_neg_branch = where_op(y_neg, base_minus_pi, base_plus_pi);
    auto by_x = where_op(x_pos, base, x_neg_branch);

    // x == 0 special cases — pick by the sign of y.
    auto by_y_neg = where_op(y_neg, neg_half_pi, zero_out);
    auto by_y_pos = where_op(y_pos, half_pi, by_y_neg);
    auto x0_branch = where_op(y_zero, zero_out, by_y_pos);

    return where_op(x_is_zero, x0_branch, by_x);
}

namespace {

// ``hypot`` and ``logaddexp`` are defined on real floating values only, as
// in the reference framework.  An integer or bool input used to fail deep
// inside the composition — a ``DtypeMismatch`` from an internal ``mul`` or
// ``add`` against a float constant — which named neither the op nor the
// reason.
void require_floating(const char* op, const TensorImplPtr& a, const TensorImplPtr& b) {
    for (const auto* t : {&a, &b}) {
        if (*t && !is_floating_point((*t)->dtype()))
            ErrorBuilder(op).not_implemented(std::string("not implemented for ") +
                                             std::string(dtype_name((*t)->dtype())) +
                                             "; cast to a floating dtype first");
    }
}

}  // namespace

// Pythagorean length, scaled so it cannot overflow on the way.
//
// The naive ``sqrt(a² + b²)`` squares first, so ``hypot(1e200, 1e200)``
// overflowed to inf while the answer, 1.41e200, is an ordinary double.
// Avoiding exactly that is what distinguishes ``hypot`` from writing the
// formula out, and the previous comment here said so and left it.
//
// Factoring out the larger magnitude gives ``m · sqrt(1 + (n/m)²)`` where
// the ratio is in [0, 1], so nothing squares above 2.  ``logaddexp``
// immediately below already does the same trick for the same reason.
//
// ``m == 0`` means both operands are zero and the answer is zero; the
// division would be 0/0, so the ratio is taken against a max with the
// smallest normal and the result multiplied by ``m`` puts it back.
TensorImplPtr hypot_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("hypot").fail("null input");
    require_floating("hypot", a_in, b_in);
    const auto [a_dev, b_dev] = detail::align_binary_devices(a_in, b_in, "hypot");
    const AmpScope amp{"hypot", a_dev->device()};
    const auto [a, b] = cast_operands(amp, a_dev, b_dev);
    auto abs_a = abs_op(a);
    auto abs_b = abs_op(b);
    auto big = maximum_op(abs_a, abs_b);
    auto small = minimum_op(abs_a, abs_b);
    auto safe = maximum_op(big, full_like_op(big, std::numeric_limits<double>::min()));
    auto ratio = div_op(small, safe);
    auto one = full_like_op(ratio, 1.0);
    return mul_op(big, sqrt_op(add_op(one, square_op(ratio))));
}

// Numerically-stable ``log(exp(a) + exp(b))``.  Factor out the per-pair max
// so only the smaller-magnitude exponential sees rounding error.
TensorImplPtr logaddexp_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("logaddexp").fail("null input");
    require_floating("logaddexp", a_in, b_in);
    const auto [a_dev, b_dev] = detail::align_binary_devices(a_in, b_in, "logaddexp");
    const AmpScope amp{"logaddexp", a_dev->device()};
    const auto [a, b] = cast_operands(amp, a_dev, b_dev);
    auto m = maximum_op(a, b);
    auto exp_a = exp_op(sub_op(a, m));
    auto exp_b = exp_op(sub_op(b, m));
    return add_op(m, log_op(add_op(exp_a, exp_b)));
}

// Tolerance check via the standard formula.  We materialise the ``rtol``
// scalar as a same-shape constant tensor so the multiply doesn't depend on
// any scalar-broadcast support in the binary ops.
TensorImplPtr
isclose_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in, double rtol, double atol) {
    if (!a_in || !b_in)
        ErrorBuilder("isclose").fail("null input");
    const auto [a_dev, b_dev] = detail::align_binary_devices(a_in, b_in, "isclose");
    const AmpScope amp{"isclose", a_dev->device()};
    const auto [a, b] = cast_operands(amp, a_dev, b_dev);
    TensorImplPtr diff;
    TensorImplPtr abs_b;
    if (!is_floating_point(a->dtype()) && !is_floating_point(b->dtype())) {
        // The tolerances are fractions.  Built in an integer dtype they
        // truncate to 0 — ``isclose`` quietly became equality — and in bool
        // ``full_like(…, 1e-8)`` is True, so every pair of bools was "close".
        // Take the difference exactly in an integer type, then compare in
        // float32 the way a floating input would.
        auto as_int = [](const TensorImplPtr& t) {
            return t->dtype() == Dtype::Bool ? astype_op(t, Dtype::I32) : t;
        };
        const auto ai = as_int(a);
        const auto bi = as_int(b);
        diff = astype_op(abs_op(sub_op(ai, bi)), Dtype::F32);
        abs_b = astype_op(abs_op(bi), Dtype::F32);
    } else {
        diff = abs_op(sub_op(a, b));
        abs_b = abs_op(b);
    }
    auto rtol_t = full_like_op(abs_b, rtol);
    auto scaled = mul_op(rtol_t, abs_b);
    auto atol_t = full_like_op(scaled, atol);
    auto tol = add_op(atol_t, scaled);
    return less_equal_op(diff, tol);
}

}  // namespace lucid
