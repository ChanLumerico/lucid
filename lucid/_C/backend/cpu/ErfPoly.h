// lucid/_C/backend/cpu/ErfPoly.h
//
// erf of a float32, branch-free so the loop around it vectorises.
//
// Accelerate has no vector erf, and libm's ``erff`` is a scalar call that
// keeps any loop containing it scalar.  The exact GELU spent most of its
// time there (Linear CHA-9): 2.2 ms forward + backward on 524k elements,
// against the reference's 0.63.  This evaluates erf in double from two
// polynomials, each a Chebyshev fit converted to monomial form:
//
//   |x| <= 1   x * P(2x^2 - 1)          P fits erf(x)/x over x^2 in [0, 1]
//   |x| >  1   sign(x) * Q((2t - 5)/3)  Q fits erf(t) over t in [1, 4]
//
// Beyond |x| = 4 the argument is clamped: erf(4) is 1 - 1.5e-8, which
// rounds to 1 in float32, as every larger value does.  The fits' error is
// relative 1e-12 below 1 and absolute 1e-11 above, thousands of times under
// float32's half-ulp, so rounding the double result gives the correctly
// rounded erf almost everywhere.  Checked over 126.6M float32 inputs evenly
// spaced in bit pattern across [-4.5, 4.5]: 1,254 (0.001%) are 1 ulp off the
// correctly rounded value and the rest are exact.  NaN propagates, +-inf
// give +-1, and -0 gives -0.
//
// The coefficients were fitted with numpy (Chebyshev interpolation at 6,000
// nodes, then ``cheb2poly``).  The fit and the check are in vault
// ``perf-cpu-gelu-erf``.  float64 keeps ``std::erf``.

#pragma once

#include <algorithm>
#include <cmath>

namespace lucid::backend::cpu {

inline float erf_f32(float xf) noexcept {
    constexpr double kSmall[] = {
        0.9654687386698628,     -0.1405360890155273,    0.019852496688099335,
        -0.0022854856569769046, 0.0002175171631475766,  -1.7536824632497327e-05,
        1.223362766628454e-06,  -7.557496500089848e-08, 4.1386945747406505e-09,
    };
    constexpr double kLarge[] = {
        0.9995930479756963,     0.003267426299726116,  -0.012252847423028367,
        0.028181554643405986,   -0.04365081934480178,  0.04645357811214869,
        -0.03187596613847961,   0.009267782635310444,  0.0066768573080467145,
        -0.009622182233519854,  0.0045585205728528215, 0.00044344761475586115,
        -0.0018600132434256227, 0.0009188632958344141, 0.00010030599042518084,
        -0.0002938107891702501, 7.670282247258007e-05, 3.293624261099777e-05,
        -1.539976011699819e-05,
    };
    const double x = xf;
    const double t = std::fabs(x);

    const double s = 2.0 * x * x - 1.0;
    double small = kSmall[8];
    for (int k = 7; k >= 0; --k)
        small = small * s + kSmall[k];
    small *= x;

    // (2t - 5) / 3, multiplied rather than divided: a vector divide costs
    // several multiplies, and the 1e-16 difference is lost in the rounding.
    constexpr double kThird = 1.0 / 3.0;
    const double u = (2.0 * std::min(t, 4.0) - 5.0) * kThird;
    double large = kLarge[18];
    for (int k = 17; k >= 0; --k)
        large = large * u + kLarge[k];
    large = std::copysign(large, x);

    return static_cast<float>(t <= 1.0 ? small : large);
}

}  // namespace lucid::backend::cpu
