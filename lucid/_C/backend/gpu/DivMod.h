// lucid/_C/backend/gpu/DivMod.h
//
// The Metal kernels of the floor-division family: ``//``, ``%`` and
// ``fmod`` on equal-shape operands, exactly, in the operands' own dtype.
// The rule is in ../DivMod.h; these are its steps in MLX ops, fused into
// one kernel per op.
//
// What MLX gives, measured on 0.32 (LCD-159):
//
//   * ``floor_divide`` on integers does not round one way across builds.
//     pip MLX 0.32.0 truncates (``-7 // 2`` is -3, eager or compiled, on
//     both streams); the vendored 0.32.3 Lucid links floors.  So the
//     integer kernel below takes whichever quotient MLX gives, applies the
//     floor correction (a no-op on a quotient that already floors), and
//     derives ``fmod`` from the floored remainder rather than from MLX's
//     quotient.
//   * ``remainder`` on integers is floored and exact at int64 (2**62 + 3
//     mod 7 is 0; arange(2e7) mod 3 is exact).
//   * ``remainder`` on floats is ``fmod`` with the floored sign fix, and
//     Metal's ``fmod`` is exact.  ``|a| mod |b|`` therefore is the exact
//     ``fmod(|a|, |b|)``, which is how ``fmod`` is built here.
//   * Metal flushes float32 subnormals to zero, so an operand or result
//     below 2**-126 differs from the CPU (``1e-40 % 3e-41`` is NaN).
//   * A kernel ``mx::compile`` fuses is built for fast math: inside one,
//     ``remainder(5, inf)`` is NaN where the plain op answers 5.  So the
//     float path runs op by op, unfused; only the integer path, which fast
//     math cannot touch, is fused.
//
// The integer quotient's ``min / -1`` wraps on Metal (int8 -128 // -1 is
// -128, as in the reference).  The 16-bit floats are computed in float32
// and rounded once.
#pragma once

#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../DivMod.h"
#include "HalfAccumulation.h"

namespace lucid::backend {

namespace gpu_divmod_detail {

namespace mx = ::mlx::core;

// True where ``x`` has its sign bit set, -0.0 included, which a comparison
// with zero cannot see: ``1 / -0.0`` is -inf.  (A NaN reads false; nothing
// here depends on its sign.)  ``view`` would read the bit directly, but a
// shapeless compile refuses it.
inline mx::array sign_bit(const mx::array& x) {
    const mx::array zero = mx::zeros_like(x);
    const mx::array negative_zero =
        mx::logical_and(mx::equal(x, zero), mx::less(mx::reciprocal(x), zero));
    return mx::logical_or(mx::less(x, zero), negative_zero);
}

// ``r`` non-zero with its sign opposite to ``s``'s.
inline mx::array signs_differ(const mx::array& r, const mx::array& s) {
    const mx::array zero = mx::zeros_like(r);
    return mx::logical_and(mx::not_equal(r, zero),
                           mx::not_equal(mx::less(r, zero), mx::less(s, zero)));
}

// MLX's quotient, truncated or floored (see above), floored here; the
// floored remainder has the sign of ``b``, and the truncated one, which
// ``fmod`` is, the sign of ``a``.
template <DivMod M>
mx::array integer(const mx::array& a, const mx::array& b) {
    const mx::array q = mx::floor_divide(a, b);
    const mx::array r = mx::subtract(a, mx::multiply(q, b));
    const mx::array to_floor = signs_differ(r, b);
    if constexpr (M == DivMod::FloorDiv)
        return mx::subtract(q, mx::astype(to_floor, q.dtype()));
    const mx::array floored = mx::where(to_floor, mx::add(r, b), r);
    if constexpr (M == DivMod::Remainder)
        return floored;
    return mx::where(signs_differ(floored, a), mx::subtract(floored, b), floored);
}

// ``a`` and ``b`` are float32; the 16-bit floats arrive widened.
template <DivMod M>
mx::array floating(const mx::array& a, const mx::array& b) {
    const mx::array r = mx::remainder(mx::abs(a), mx::abs(b));
    const mx::array mod = mx::where(sign_bit(a), mx::negative(r), r);
    if constexpr (M == DivMod::Fmod)
        return mod;
    const mx::array zero = mx::zeros_like(mod);
    const mx::array adjust = signs_differ(mod, b);
    if constexpr (M == DivMod::Remainder)
        return mx::where(adjust, mx::add(mod, b), mod);
    const mx::array one = mx::ones_like(mod);
    const mx::array quotient = mx::divide(a, b);
    mx::array div = mx::divide(mx::subtract(a, mod), b);
    div = mx::where(adjust, mx::subtract(div, one), div);
    mx::array whole = mx::floor(div);
    whole = mx::where(mx::greater(mx::subtract(div, whole), mx::array(0.5f)), mx::add(whole, one),
                      whole);
    const mx::array signed_zero = mx::where(sign_bit(quotient), mx::negative(zero), zero);
    whole = mx::where(mx::not_equal(div, zero), whole, signed_zero);
    return mx::where(mx::equal(b, zero), quotient, whole);
}

template <DivMod M>
struct Steps {
    std::vector<mx::array> operator()(const std::vector<mx::array>& in) const {
        const mx::Dtype dt = in[0].dtype();
        if (!mx::issubdtype(dt, mx::floating))
            return {integer<M>(in[0], in[1])};
        return {narrow_to(floating<M>(widen_half(in[0]), widen_half(in[1])), dt)};
    }
};

}  // namespace gpu_divmod_detail

namespace gpu_divmod_detail {

template <DivMod M>
mx::array run(const mx::array& a, const mx::array& b) {
    if (mx::issubdtype(a.dtype(), mx::floating))
        return Steps<M>{}({a, b})[0];
    return mlx_fused({a, b}, Steps<M>{})[0];
}

}  // namespace gpu_divmod_detail

// ``a (op) b`` element-wise; ``a`` and ``b`` have one shape and one dtype.
inline ::mlx::core::array
gpu_divmod(DivMod mode, const ::mlx::core::array& a, const ::mlx::core::array& b) {
    switch (mode) {
    case DivMod::FloorDiv:
        return gpu_divmod_detail::run<DivMod::FloorDiv>(a, b);
    case DivMod::Remainder:
        return gpu_divmod_detail::run<DivMod::Remainder>(a, b);
    case DivMod::Fmod:
        break;
    }
    return gpu_divmod_detail::run<DivMod::Fmod>(a, b);
}

// The floored quotient of float32 ``a`` and ``b``, broadcast.  Never call it
// inside ``mx::compile`` (see the fast-math note above).
inline ::mlx::core::array gpu_floor_divide_float32(const ::mlx::core::array& a,
                                                   const ::mlx::core::array& b) {
    return gpu_divmod_detail::floating<DivMod::FloorDiv>(a, b);
}

// Whether any element of the integer array ``b`` is zero.  Reads the
// answer back to the host, so it waits for ``b``.
inline bool gpu_any_zero(const ::mlx::core::array& b) {
    namespace mx = ::mlx::core;
    return mx::any(mx::equal(b, mx::zeros_like(b))).item<bool>();
}

}  // namespace lucid::backend
