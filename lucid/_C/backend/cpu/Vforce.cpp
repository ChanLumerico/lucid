// lucid/_C/backend/cpu/Vforce.cpp
//
// Implements the vForce wrappers declared in Vforce.h.  All non-power functions
// hand off to the corresponding vv*f / vv* symbol from Accelerate.  The count
// argument required by vForce is a signed int*, so every function captures a
// local int before the call via the helper N().
//
// vpow_f32 / vpow_f64 note: vvpowf and vvpow take (out, exponent, base, &n),
// which is the opposite order from pow(base, exp) in C.  The wrappers accept
// (base, expo, out, n) to match the Lucid calling convention and internally
// pass expo before base to the vForce API.
//
// The LUCID_VFORCE_UNARY macro at the bottom expands pairs of f32/f64 wrappers
// for functions that share a uniform (out, in, &count) signature (log2, fabs,
// floor, ceil, round).  Trigonometric and hyperbolic functions use
// LUCID_LIBM_UNARY over <cmath> (LCD-472): Accelerate's vvcosf/vvcos return
// 1 - 1 ulp at 0, vvtanf/vvtan lose the negative sign of -0.0, and the vForce
// trig/hyperbolic entries have 2-3 ulp errors on Gaussian inputs.
//
// Square root and reciprocal are IEEE operations the reference rounds
// correctly, and the float vForce entries for them do not, so their float
// wrappers are plain loops (see vsqrt_f32 and vrec_f32).

#include "Vforce.h"

#include <cmath>

#include <Accelerate/Accelerate.h>

#include "ErfPoly.h"
#include "Parallel.h"

namespace lucid::backend::cpu {

namespace {
// Converts a std::size_t element count to the signed int* expected by all
// vForce vector math functions.
inline int N(std::size_t n) {
    return static_cast<int>(n);
}

// Elements of a vForce call worth a core of their own: at a few ns an
// element, 16k of them are tens of microseconds, well past a dispatch.
constexpr std::size_t kGrain = 16384;

// Runs ``fn(out, in, &count)`` over [0, n) in chunks across cores.  Every
// vForce function here is element-wise, so a chunk computes exactly what the
// whole call would have — the bits do not change.  One call on one core was
// what kept ``lucid.exp`` at twice the reference and ``lucid.erf`` at 14x.
// A call from inside a parallel chunk (GELU's tiles) is below the grain and
// runs inline.
template <class T, class F>
inline void split(const T* in, T* out, std::size_t n, F&& fn) {
    parallel_for(n, kGrain, [&](std::size_t lo, std::size_t hi) {
        int count = N(hi - lo);
        fn(out + lo, in + lo, &count);
    });
}

// ``out[i] = fn(in[i])`` over [0, n), chunked across cores like split().
// Used where the vForce entry is not exact, and for erf, which vForce lacks.
template <class T, class F>
inline void split_map(const T* in, T* out, std::size_t n, F fn) {
    parallel_for(n, kGrain, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i)
            out[i] = fn(in[i]);
    });
}
}  // namespace

void vexp_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvexpf(o, i, c); });
}

void vlog_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvlogf(o, i, c); });
}

// Deliberately not vvsqrtf, which is not correctly rounded: on 1e6 gaussian
// magnitudes 41% of its roots were 1–2 ulp off the IEEE root.  The
// reference's square root is exact, and so is the one on Metal.  This loop
// compiles to NEON ``fsqrt``, which IEEE 754 requires to be correctly
// rounded, and it measured faster on one core (0.17 ms against 0.31 ms on
// 1e6 elements).  The double path stays on ``vvsqrt``, which was exact on the
// same data.
void vsqrt_f32(const float* in, float* out, std::size_t n) {
    split_map(in, out, n, [](float x) { return std::sqrt(x); });
}

void vexp_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvexp(o, i, c); });
}

void vlog_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvlog(o, i, c); });
}

void vsqrt_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvsqrt(o, i, c); });
}

void vpow_f32(const float* base, const float* expo, float* out, std::size_t n) {
    int count = N(n);

    vvpowf(out, expo, base, &count);
}

void vpow_f64(const double* base, const double* expo, double* out, std::size_t n) {
    int count = N(n);
    vvpow(out, expo, base, &count);
}

// Trigonometric and hyperbolic functions use libm (<cmath>) rather than
// vForce (LCD-472): vvcosf(0) = 0.99999994 and vvcos(0) = 0.9999999999999999
// (1 ulp below 1.0), vvtanf(-0.0) and vvtan(-0.0) return +0.0, and the vForce
// trig/hyperbolic entries have 2-3 ulp errors on Gaussian inputs.  libm is
// exact at +-0.0 and within 1 ulp across both widths.
#define LUCID_LIBM_UNARY(NAME, FN)                                                                 \
    void NAME##_f32(const float* in, float* out, std::size_t n) {                                  \
        split_map(in, out, n, [](float x) { return std::FN(x); });                                 \
    }                                                                                              \
    void NAME##_f64(const double* in, double* out, std::size_t n) {                                \
        split_map(in, out, n, [](double x) { return std::FN(x); });                                \
    }

LUCID_LIBM_UNARY(vsin, sin)
LUCID_LIBM_UNARY(vcos, cos)
LUCID_LIBM_UNARY(vtan, tan)
LUCID_LIBM_UNARY(vasin, asin)
LUCID_LIBM_UNARY(vacos, acos)
LUCID_LIBM_UNARY(vatan, atan)
LUCID_LIBM_UNARY(vsinh, sinh)
LUCID_LIBM_UNARY(vcosh, cosh)
LUCID_LIBM_UNARY(vtanh, tanh)

#undef LUCID_LIBM_UNARY

// Expands a matching f32 and f64 wrapper pair for any vForce function that
// follows the (output_ptr, input_ptr, &count) calling convention.
#define LUCID_VFORCE_UNARY(NAME, F32, F64)                                                         \
    void NAME##_f32(const float* in, float* out, std::size_t n) {                                  \
        split(in, out, n, [](float* o, const float* i, int* c) { F32(o, i, c); });                 \
    }                                                                                              \
    void NAME##_f64(const double* in, double* out, std::size_t n) {                                \
        split(in, out, n, [](double* o, const double* i, int* c) { F64(o, i, c); });               \
    }

LUCID_VFORCE_UNARY(vlog2, vvlog2f, vvlog2)
LUCID_VFORCE_UNARY(vfabs, vvfabsf, vvfabs)
LUCID_VFORCE_UNARY(vfloor, vvfloorf, vvfloor)
LUCID_VFORCE_UNARY(vceil, vvceilf, vvceil)
LUCID_VFORCE_UNARY(vround, vvnintf, vvnint)

#undef LUCID_VFORCE_UNARY

// Deliberately not vvrecf, which is not correctly rounded: on 1e6 gaussian
// values 34% of its reciprocals were 1 ulp off ``1 / x``.  The reference
// divides, and sigmoid, rsqrt and the half reciprocals all go through here.
// ``fdiv`` is exact and measured faster on one core (0.085 ms against
// 0.15 ms on 1e6 elements).  ``vvrec`` was exact on the same data and stays.
void vrec_f32(const float* in, float* out, std::size_t n) {
    split_map(in, out, n, [](float x) { return 1.0f / x; });
}

void vrec_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](double* o, const double* i, int* c) { vvrec(o, i, c); });
}

// erf — Apple Accelerate does not expose a vForce erf symbol.  float32 uses
// erf_f32 (ErfPoly.h): within 1 ulp of the correctly rounded value and
// branch-free, so this loop vectorises, which a loop over libm's erff did
// not.  float64 keeps std::erf.
void verf_f32(const float* in, float* out, std::size_t n) {
    parallel_for(n, kGrain, [&](std::size_t lo, std::size_t hi) {
        const float* __restrict x = in;
        float* __restrict y = out;
        for (std::size_t i = lo; i < hi; ++i)
            y[i] = erf_f32(x[i]);
    });
}
void verf_f64(const double* in, double* out, std::size_t n) {
    split_map(in, out, n, [](double x) { return std::erf(x); });
}

}  // namespace lucid::backend::cpu
