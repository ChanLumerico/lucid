// lucid/_C/backend/cpu/GeluGrad.h
//
// The exact GELU's derivative, Phi(x) + x * phi(x), for float32 on the CPU:
// Phi the Gaussian CDF (through erf_f32, ErfPoly.h) and phi its density.
//
// The density used to come from vForce's expf, called on an L1 tile between
// two passes over it -- one writing -x^2/2, one doing the erf and the sum.
// The call in the middle of the tile loop cost more than the exp itself:
// vector registers do not survive a call, so the compiler kept the erf's
// constants in integer registers and rebuilt each one with a dup on every
// iteration of the second pass -- ~0.7 ns an element, where the expf is
// 0.44.  The loop ran 2.4x the forward's per element (Linear CHA-25).
// normal_pdf_f32 below is an inline float exp, branch-free like erf_f32, so
// neither pass makes a call.  The two passes stay separate functions that
// are never inlined: the erf's constants and the exp's together outnumber
// the 32 vector registers, and one fused loop spilled them.  Per element on
// one core: 1.33 ns for the two passes, 1.50 fused, 1.99 for the expf
// route; the forward is 0.81.
//
// The new density is also more accurate than the expf route.  That route
// rounded x^2 before the exp, an error the exp multiplies by x^2/2 -- up to
// 65 ulp of the density for |x| in [8, 13).  Here the rounding of x^2 is
// recovered exactly with an fma and folded into the reduced argument, and
// the density is within 2 ulp of the correctly rounded value for every
// float (60-66% exact, band by band).  The derivative's own error -- what
// the kernel adds to the float CDF both routes share -- is no larger than
// before in any band of x, and 3 ulp where it was 66 for x in [-13, -6),
// where the CDF has rounded to zero.  Measured over every float with
// |x| in [2^-12, 32); vault ``perf-cpu-gelu-backward-fused`` has the table.

#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ErfPoly.h"

namespace lucid::backend::cpu {

// The standard normal density exp(-x^2 / 2) / sqrt(2 pi), in float32
// arithmetic, branch-free so the loop around it vectorises four lanes wide.
//
//   exp(-s) = 2^n * e^r,   s = x^2 / 2,   n = -round(s / ln 2),   |r| <= ln2 / 2
//
// s is carried as two floats: s = h * a rounded (h = a / 2) and, negated,
// what that rounding dropped, fma(-h, a, s), exact.  The low part joins the
// reduced argument together with the low half of n * ln 2 (Cody-Waite:
// n * ln2_hi is exact and s + n * ln2_hi cancels exactly), so r is rounded
// once, at its own size.  e^r is 1 + r * P(r), P of degree 5 fitted to 3e-9.
//
// 2^n is added straight into the exponent field of e^r.  Adding 1.5 * 2^23
// + 64 rounds -s / ln 2 to an integer and leaves n + 64 in the low bits, so
// the scale is a shift and an integer add, and e^r * 2^(n + 64) is a normal
// float for every n the clamp allows.  The 2^-64 is folded into the
// 1 / sqrt(2 pi) of the last multiply, so a density that underflows
// (|x| > 13.39) is rounded once, into the subnormals, instead of being
// rounded there and then multiplied again.
//
// |x| is clamped to 14.5, past the point where the density rounds to zero
// (14.36), which keeps x^2 and n in range for any input: +-inf and NaN give
// 0, as huge finite inputs do.
inline float normal_pdf_f32(float x) noexcept {
    // (e^r - 1) / r, degree 5.
    constexpr float kP[] = {1.0f, 0.5f, 0.1666652f, 0.041666403f, 0.008368823f, 0.0013940529f};
    constexpr float kLog2e = 1.44269504f;
    constexpr float kShift = 12582976.0f;  // 1.5 * 2^23 + 64
    constexpr float kLn2Hi = 0.693145751953125f;
    constexpr float kLn2Lo = 1.42860677e-06f;
    constexpr float kScaledInvSqrt2Pi = 0.3989422804014327f * 0x1p-64f;
    constexpr float kClamp = 14.5f;

    const float a = std::fmin(std::fabs(x), kClamp);
    const float h = 0.5f * a;
    const float s = h * a;
    const float s_lo_neg = std::fma(-h, a, s);      // -(a^2 / 2 - s), exactly
    const float sh = std::fma(-s, kLog2e, kShift);  // 1.5 * 2^23 + 64 + n
    const float n = sh - kShift;
    const float r_hi_neg = std::fma(n, kLn2Hi, s);  // exact
    const float r = std::fma(n, -kLn2Lo, s_lo_neg) - r_hi_neg;
    float p = kP[5];
    for (int k = 4; k >= 0; --k)
        p = std::fma(p, r, kP[k]);
    const float e = std::fma(r, p, 1.0f);
    const std::uint32_t bits =
        std::bit_cast<std::uint32_t>(e) + (std::bit_cast<std::uint32_t>(sh) << 23);
    return std::bit_cast<float>(bits) * kScaledInvSqrt2Pi;
}

// First pass over a tile: cdf[i] = Phi(x[i]) = (1 + erf(x[i] / sqrt 2)) / 2.
//
// Never inlined, so its loop holds the erf's constants in registers and
// nothing else (see the header note).
[[gnu::noinline]] inline void
gelu_exact_cdf_f32(const float* __restrict x, float* __restrict cdf, std::size_t n) noexcept {
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (std::size_t i = 0; i < n; ++i)
        cdf[i] = 0.5f * (1.f + erf_f32(x[i] * kInvSqrt2));
}

// Second pass: out[i] = (cdf[i] + x[i] * phi(x[i])) * g[i].
//
// Every operation is a single explicit IEEE one, so the vectorised body and
// the scalar tail give an element the same bits.
[[gnu::noinline]] inline void gelu_exact_grad_f32(const float* __restrict x,
                                                  const float* __restrict cdf,
                                                  const float* __restrict g,
                                                  float* __restrict out,
                                                  std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const float xi = x[i];
        out[i] = std::fma(xi, normal_pdf_f32(xi), cdf[i]) * g[i];
    }
}

}  // namespace lucid::backend::cpu
