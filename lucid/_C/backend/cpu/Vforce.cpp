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
// for functions that share a uniform (out, in, &count) signature (asin, acos,
// atan, sinh, cosh, log2, fabs, rec, floor, ceil, round).

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
}  // namespace

void vexp_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvexpf(o, i, c); });
}

void vlog_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvlogf(o, i, c); });
}

void vsqrt_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvsqrtf(o, i, c); });
}

void vtanh_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvtanhf(o, i, c); });
}

void vsin_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvsinf(o, i, c); });
}

void vcos_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvcosf(o, i, c); });
}

void vtan_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvtanf(o, i, c); });
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

void vtanh_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvtanh(o, i, c); });
}

void vsin_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvsin(o, i, c); });
}

void vcos_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvcos(o, i, c); });
}

void vtan_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvtan(o, i, c); });
}

void vpow_f32(const float* base, const float* expo, float* out, std::size_t n) {
    int count = N(n);

    vvpowf(out, expo, base, &count);
}

void vpow_f64(const double* base, const double* expo, double* out, std::size_t n) {
    int count = N(n);
    vvpow(out, expo, base, &count);
}

// Expands a matching f32 and f64 wrapper pair for any vForce function that
// follows the (output_ptr, input_ptr, &count) calling convention.
#define LUCID_VFORCE_UNARY(NAME, F32, F64)                                                         \
    void NAME##_f32(const float* in, float* out, std::size_t n) {                                  \
        split(in, out, n, [](float* o, const float* i, int* c) { F32(o, i, c); });                 \
    }                                                                                              \
    void NAME##_f64(const double* in, double* out, std::size_t n) {                                \
        split(in, out, n, [](double* o, const double* i, int* c) { F64(o, i, c); });               \
    }

LUCID_VFORCE_UNARY(vasin, vvasinf, vvasin)
LUCID_VFORCE_UNARY(vacos, vvacosf, vvacos)
LUCID_VFORCE_UNARY(vatan, vvatanf, vvatan)
LUCID_VFORCE_UNARY(vsinh, vvsinhf, vvsinh)
LUCID_VFORCE_UNARY(vcosh, vvcoshf, vvcosh)
LUCID_VFORCE_UNARY(vlog2, vvlog2f, vvlog2)
LUCID_VFORCE_UNARY(vfabs, vvfabsf, vvfabs)
LUCID_VFORCE_UNARY(vrec, vvrecf, vvrec)
LUCID_VFORCE_UNARY(vfloor, vvfloorf, vvfloor)
LUCID_VFORCE_UNARY(vceil, vvceilf, vvceil)
LUCID_VFORCE_UNARY(vround, vvnintf, vvnint)

#undef LUCID_VFORCE_UNARY

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
    parallel_for(n, kGrain, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i)
            out[i] = std::erf(in[i]);
    });
}

}  // namespace lucid::backend::cpu
