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

#include <Accelerate/Accelerate.h>

namespace lucid::backend::cpu {

namespace {
// Converts a std::size_t element count to the signed int* expected by all
// vForce vector math functions.
inline int N(std::size_t n) {
    return static_cast<int>(n);
}
}  // namespace

void vexp_f32(const float* in, float* out, std::size_t n) {
    int count = N(n);
    vvexpf(out, in, &count);
}

void vlog_f32(const float* in, float* out, std::size_t n) {
    int count = N(n);
    vvlogf(out, in, &count);
}

void vsqrt_f32(const float* in, float* out, std::size_t n) {
    int count = N(n);
    vvsqrtf(out, in, &count);
}

void vtanh_f32(const float* in, float* out, std::size_t n) {
    int count = N(n);
    vvtanhf(out, in, &count);
}

void vsin_f32(const float* in, float* out, std::size_t n) {
    int count = N(n);
    vvsinf(out, in, &count);
}

void vcos_f32(const float* in, float* out, std::size_t n) {
    int count = N(n);
    vvcosf(out, in, &count);
}

void vtan_f32(const float* in, float* out, std::size_t n) {
    int count = N(n);
    vvtanf(out, in, &count);
}

void vexp_f64(const double* in, double* out, std::size_t n) {
    int count = N(n);
    vvexp(out, in, &count);
}

void vlog_f64(const double* in, double* out, std::size_t n) {
    int count = N(n);
    vvlog(out, in, &count);
}

void vsqrt_f64(const double* in, double* out, std::size_t n) {
    int count = N(n);
    vvsqrt(out, in, &count);
}

void vtanh_f64(const double* in, double* out, std::size_t n) {
    int count = N(n);
    vvtanh(out, in, &count);
}

void vsin_f64(const double* in, double* out, std::size_t n) {
    int count = N(n);
    vvsin(out, in, &count);
}

void vcos_f64(const double* in, double* out, std::size_t n) {
    int count = N(n);
    vvcos(out, in, &count);
}

void vtan_f64(const double* in, double* out, std::size_t n) {
    int count = N(n);
    vvtan(out, in, &count);
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
        int c = N(n);                                                                              \
        F32(out, in, &c);                                                                          \
    }                                                                                              \
    void NAME##_f64(const double* in, double* out, std::size_t n) {                                \
        int c = N(n);                                                                              \
        F64(out, in, &c);                                                                          \
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

}  // namespace lucid::backend::cpu
