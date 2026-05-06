// lucid/_C/ops/composite/Math.cpp
//
// Composition-based elementwise math ops.  Each entry function builds its
// forward graph from existing differentiable ``*_op`` calls; the chain rule
// is provided automatically by the primitives' backward nodes (LogBackward,
// ExpBackward, MulBackward, etc.) so no new schemas are registered here.

#include "Math.h"

#include <cmath>

#include "../../core/TensorImpl.h"
#include "../bfunc/Add.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Div.h"
#include "../bfunc/Maximum.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Arith.h"
#include "../ufunc/Discrete.h"
#include "../ufunc/Exponential.h"
#include "../ufunc/Trig.h"
#include "../utils/Select.h"

namespace lucid {

// ── unary compositions ──────────────────────────────────────────────────────

// ``log10(x) = log(x) / log(10)``.  Materialises ``log(10)`` as a same-shape
// constant so the divide kernel does not need scalar broadcasting.
TensorImplPtr log10_op(const TensorImplPtr& a) {
    auto log_a = log_op(a);
    auto ln10 = full_like_op(log_a, std::log(10.0));
    return div_op(log_a, ln10);
}

// ``log1p(x) = log(1 + x)``.  No fused kernel exists, so this composes; the
// numerical accuracy near x = 0 is therefore that of the underlying ``log``.
TensorImplPtr log1p_op(const TensorImplPtr& a) {
    auto ones = full_like_op(a, 1.0);
    return log_op(add_op(a, ones));
}

// ``exp2(x) = exp(x · log(2))``.  Multiplying against a constant tensor keeps
// the gradient chain visible to ``MulBackward`` and ``ExpBackward``.
TensorImplPtr exp2_op(const TensorImplPtr& a) {
    auto ln2 = full_like_op(a, std::log(2.0));
    return exp_op(mul_op(a, ln2));
}

// Round toward zero — pick ``floor`` for non-negative inputs and ``ceil`` for
// negatives.  Both branches are piecewise constant, so the backward chain
// contributes zero gradient (matching the reference framework).
TensorImplPtr trunc_op(const TensorImplPtr& a) {
    auto zero = full_like_op(a, 0.0);
    auto cond = greater_equal_op(a, zero);
    return where_op(cond, floor_op(a), ceil_op(a));
}

// Fractional part — the ``trunc`` term contributes zero gradient, so the
// gradient w.r.t. ``a`` is just the upstream gradient.
TensorImplPtr frac_op(const TensorImplPtr& a) {
    return sub_op(a, trunc_op(a));
}

// ── binary compositions ─────────────────────────────────────────────────────

// Quadrant-aware arctangent.  Patches ``arctan(y / x)`` with ±π/zero/±π/2
// branches via ``where`` based on the signs of the operands; the divide is
// guarded by substituting 1 for x = 0 so the safe branch is always defined.
TensorImplPtr atan2_op(const TensorImplPtr& y, const TensorImplPtr& x) {
    constexpr double pi = 3.14159265358979323846;

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

// ``fmod(a, b) = a − trunc(a / b) · b``.  C-style modulo: result has the
// sign of ``a`` because ``trunc`` rounds toward zero.
TensorImplPtr fmod_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto q = div_op(a, b);
    auto k = trunc_op(q);
    return sub_op(a, mul_op(k, b));
}

// ``remainder(a, b) = a − floor(a / b) · b``.  Python-style modulo: result
// has the sign of ``b`` because ``floor`` rounds toward −∞.
TensorImplPtr remainder_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto q = div_op(a, b);
    auto k = floor_op(q);
    return sub_op(a, mul_op(k, b));
}

// Pythagorean length.  Naive ``sqrt(a² + b²)`` — does not perform IEEE-754
// overflow-safe scaling, matching the engine's ``square_op`` precision.
TensorImplPtr hypot_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return sqrt_op(add_op(square_op(a), square_op(b)));
}

// Numerically-stable ``log(exp(a) + exp(b))``.  Factor out the per-pair max
// so only the smaller-magnitude exponential sees rounding error.
TensorImplPtr logaddexp_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    auto m = maximum_op(a, b);
    auto exp_a = exp_op(sub_op(a, m));
    auto exp_b = exp_op(sub_op(b, m));
    return add_op(m, log_op(add_op(exp_a, exp_b)));
}

// Tolerance check via the standard formula.  We materialise the ``rtol``
// scalar as a same-shape constant tensor so the multiply doesn't depend on
// any scalar-broadcast support in the binary ops.
TensorImplPtr
isclose_op(const TensorImplPtr& a, const TensorImplPtr& b, double rtol, double atol) {
    auto diff = abs_op(sub_op(a, b));
    auto abs_b = abs_op(b);
    auto rtol_t = full_like_op(abs_b, rtol);
    auto scaled = mul_op(rtol_t, abs_b);
    auto atol_t = full_like_op(scaled, atol);
    auto tol = add_op(atol_t, scaled);
    return less_equal_op(diff, tol);
}

}  // namespace lucid
