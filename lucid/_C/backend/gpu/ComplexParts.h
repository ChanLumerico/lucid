// lucid/_C/backend/gpu/ComplexParts.h
//
// A complex64 array taken apart into float32 parts and put back together,
// for the GPU work MLX has no complex64 kernel for (its scatter-add family)
// or judges the complex value as a whole where each part counts on its own
// (``nan_to_num``).
#pragma once

#include <cstdint>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../core/Half.h"

namespace lucid {
namespace backend {

// A complex64 array of ``re``'s shape from float32 parts of that shape.
// Assembling it as ``re + im * 1j`` turns an infinite part into NaN — the
// product computes ``inf * 0`` for the other lane, so ``complex(inf, inf)``
// answered (nan + inf j) and a scatter-add of (1 + inf j) did too.  A
// complex64 is two adjacent float32s, so the parts are interleaved and the
// pairs reinterpreted: no arithmetic touches either one.
inline ::mlx::core::array complex_from_parts(const ::mlx::core::array& re,
                                             const ::mlx::core::array& im) {
    namespace mx = ::mlx::core;
    auto pairs = mx::contiguous(mx::stack({re, im}, -1));
    return mx::reshape(mx::view(pairs, mx::complex64), re.shape());
}

// ``x`` with NaN, +inf and -inf replaced, for a real float ``x``.
inline ::mlx::core::array replace_nonfinite(const ::mlx::core::array& x,
                                            double nan_val,
                                            double posinf_val,
                                            double neginf_val) {
    namespace mx = ::mlx::core;
    const auto mdt = x.dtype();
    if (mdt == mx::bfloat16) {
        // MLX's bfloat16 ``where`` routes through float32 conversion on Metal,
        // which flushes bfloat16 subnormals to zero; replace over the 16-bit
        // IEEE words so subnormals and signed zeros survive bit-for-bit.
        auto bits = mx::view(mx::contiguous(x), mx::uint16);
        auto mag = mx::bitwise_and(bits, mx::array(static_cast<std::uint16_t>(0x7FFFu)));
        const auto exp_mask = mx::array(static_cast<std::uint16_t>(0x7F80u));
        const auto nan_b = mx::array(detail::float_to_bfloat16_bits(static_cast<float>(nan_val)));
        const auto pi_b = mx::array(detail::float_to_bfloat16_bits(static_cast<float>(posinf_val)));
        const auto ni_b = mx::array(detail::float_to_bfloat16_bits(static_cast<float>(neginf_val)));
        auto inf_b = mx::where(mx::equal(bits, mag), pi_b, ni_b);
        auto kept = mx::where(mx::equal(mag, exp_mask), inf_b, bits);
        auto out_bits = mx::where(mx::greater(mag, exp_mask), nan_b, kept);
        return mx::view(out_bits, mx::bfloat16);
    }
    mx::array nan_a(static_cast<float>(nan_val), mdt);
    mx::array pi_a(static_cast<float>(posinf_val), mdt);
    mx::array ni_a(static_cast<float>(neginf_val), mdt);
    auto out = mx::where(mx::isnan(x), nan_a, x);
    auto pos_inf = mx::logical_and(mx::isinf(out), mx::greater(out, mx::array(0.f, mdt)));
    out = mx::where(pos_inf, pi_a, out);
    auto neg_inf = mx::logical_and(mx::isinf(out), mx::less(out, mx::array(0.f, mdt)));
    return mx::where(neg_inf, ni_a, out);
}

// ``op(base, updates)`` for complex64 operands, run on the real parts and on
// the imaginary parts.  Right for an operation that acts on each part alone:
// a sum, or an overwrite.
template <class Op>
::mlx::core::array
by_complex_parts(const ::mlx::core::array& base, const ::mlx::core::array& updates, Op&& op) {
    namespace mx = ::mlx::core;
    return complex_from_parts(op(mx::real(base), mx::real(updates)),
                              op(mx::imag(base), mx::imag(updates)));
}

}  // namespace backend
}  // namespace lucid
