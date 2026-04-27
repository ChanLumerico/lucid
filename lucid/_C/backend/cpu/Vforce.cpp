#include "Vforce.h"

#include <Accelerate/Accelerate.h>

namespace lucid::backend::cpu {

namespace {
inline int N(std::size_t n) { return static_cast<int>(n); }
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
    // vvpowf signature: vvpowf(y, expo, base, &count) — y[i] = base[i] ^ expo[i].
    vvpowf(out, expo, base, &count);
}

void vpow_f64(const double* base, const double* expo, double* out, std::size_t n) {
    int count = N(n);
    vvpow(out, expo, base, &count);
}

#define LUCID_VFORCE_UNARY(NAME, F32, F64) \
    void NAME##_f32(const float* in, float* out, std::size_t n) { \
        int c = N(n); F32(out, in, &c); \
    } \
    void NAME##_f64(const double* in, double* out, std::size_t n) { \
        int c = N(n); F64(out, in, &c); \
    }

LUCID_VFORCE_UNARY(vasin,  vvasinf,  vvasin)
LUCID_VFORCE_UNARY(vacos,  vvacosf,  vvacos)
LUCID_VFORCE_UNARY(vatan,  vvatanf,  vvatan)
LUCID_VFORCE_UNARY(vsinh,  vvsinhf,  vvsinh)
LUCID_VFORCE_UNARY(vcosh,  vvcoshf,  vvcosh)
LUCID_VFORCE_UNARY(vlog2,  vvlog2f,  vvlog2)
LUCID_VFORCE_UNARY(vfabs,  vvfabsf,  vvfabs)
LUCID_VFORCE_UNARY(vrec,   vvrecf,   vvrec)
LUCID_VFORCE_UNARY(vfloor, vvfloorf, vvfloor)
LUCID_VFORCE_UNARY(vceil,  vvceilf,  vvceil)
LUCID_VFORCE_UNARY(vround, vvnintf,  vvnint)  // banker's round (half-to-even)

#undef LUCID_VFORCE_UNARY

}  // namespace lucid::backend::cpu
