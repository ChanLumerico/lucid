// lucid/_C/backend/cpu/ErfPoly.h
//
// erf of a float32, in float32 arithmetic, branch-free so the loop around it
// vectorises four lanes wide.
//
// Accelerate has no vector erf, and libm's ``erff`` is a scalar call that
// keeps any loop containing it scalar.  The exact GELU spent most of its
// time there (Linear CHA-9).  The first replacement evaluated two
// polynomials in double to stay within 1 ulp; double runs two lanes wide and
// needed 9 + 19 terms, and the GELU stayed ~2x the reference.  This one keeps
// every operation in float and holds the same bound -- at most 1 ulp from
// the correctly rounded erf -- by arranging the arithmetic so that no
// rounding before the last one lands on a value comparable to the result:
//
//   |x| < 1   x + x * P(z),  z = x^2       P fits erf(x)/x - 1 over z in [0, 1]
//   |x| >= 1  sign(x) * (E + u * S(u))     u = min(|x|, U) - c,  S over [1, U] - c
//
// Below 1 the leading term x is exact and P is at most 0.157 in magnitude,
// so P's rounding, and z's, reach the result scaled down by 1/8 or more.
//
// From 1 up the centre c is a float chosen so that erf(c) is within 6e-14
// of a float, E: the constant term then carries no rounding, and the sum
// E + u * S is rounded once.  u itself is exact (|x| and c are both
// multiples of 2^-23 there and |u| < 2), so the variable carries no error
// either.  Above U = 3.95 every erf rounds to 1 (erf(3.92) already does);
// clamping |x| to U makes +-inf give +-1 and keeps the polynomial in range.
// S, degree 14, is evaluated as four interleaved chains in u^4 (Estrin)
// rather than one Horner chain: NEON's fma overwrites its addend, so each
// Horner step is a register copy plus an fma, all fifteen in one dependent
// line, and the split is 3-4% faster for two extra multiplies.
//
// NaN takes the |x| < 1 side (``t >= 1`` is false for it) and propagates
// through x + x * P; that leaves the clamp free to be a single fminnm.
//
// Fit error: 1.5e-9 relative below 1, 5.0e-10 absolute above.  Checked
// over every positive float32 (2.14e9 inputs) against the double-precision
// erf rounded to float32: none more than 1 ulp off, 0.69% exactly 1 ulp off
// (3.8% of inputs in [1e-4, 0.5), 7.2% in [0.5, 1), 5.1% in [1, 1.5),
// under 2% above; the double version was off on 0.001%).  The function is
// odd by construction, so negative inputs mirror the positive ones bit for
// bit.  +-inf give +-1 and -0 gives -0.
//
// The coefficients were fitted with numpy + mpmath (Lawson-weighted least
// squares, rounded to float32 one at a time with the rest refitted); the
// fit and the check are in vault ``perf-cpu-erf-float``.  float64 keeps
// ``std::erf``.

#pragma once

#include <cmath>

namespace lucid::backend::cpu {

inline float erf_f32(float x) noexcept {
    // erf(x)/x - 1 in z = x^2, degree 6.
    constexpr float kP[] = {
        0.128379166f,   -0.37612626f,    0.112835832f,   -0.0268536918f,
        0.00518807396f, -0.00080078782f, 7.8461082e-05f,
    };
    // (erf(c + u) - E) / u, degree 14.
    constexpr float kS[] = {
        0.00259734248f,  -0.00640130974f,  0.00965194125f,  -0.00976002961f,  0.00672529684f,
        -0.00292262947f, 0.000458980678f,  0.000344149506f, -0.000281126791f, 7.66864614e-05f,
        1.45548011e-05f, -1.71916108e-05f, 3.03993011e-06f, 1.07512074e-06f,  -3.59357017e-07f,
    };
    constexpr float kSplit = 1.0f;
    constexpr float kClamp = 3.95000005f;
    constexpr float kCentre = 2.46455812f;
    constexpr float kErfCentre = 0.999508619f;  // erf(kCentre), off by 6e-14

    const float t = std::fabs(x);

    const float z = x * x;
    float p = kP[6];
    for (int k = 5; k >= 0; --k)
        p = std::fma(p, z, kP[k]);
    const float small = std::fma(x, p, x);

    // S = (S0 + u S1) + u^2 (S2 + u S3), Sj(v) = sum_i kS[4i + j] v^i, v = u^4.
    const float u = std::fmin(t, kClamp) - kCentre;
    const float u2 = u * u;
    const float u4 = u2 * u2;
    float s0 = kS[12];
    float s1 = kS[13];
    float s2 = kS[14];
    float s3 = kS[11];
    for (int k = 8; k >= 0; k -= 4) {
        s0 = std::fma(s0, u4, kS[k]);
        s1 = std::fma(s1, u4, kS[k + 1]);
        s2 = std::fma(s2, u4, kS[k + 2]);
    }
    for (int k = 7; k >= 3; k -= 4)
        s3 = std::fma(s3, u4, kS[k]);
    const float s = std::fma(std::fma(s3, u, s2), u2, std::fma(s1, u, s0));
    const float large = std::copysign(std::fma(u, s, kErfCentre), x);

    return t >= kSplit ? large : small;
}

}  // namespace lucid::backend::cpu
