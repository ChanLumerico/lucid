#include "Vdsp.h"

#include <Accelerate/Accelerate.h>

namespace lucid::backend::cpu {

namespace {

// vDSP types: vDSP_Length is `unsigned long`, vDSP_Stride is `long`.
inline vDSP_Length L(std::size_t n) { return static_cast<vDSP_Length>(n); }

}  // namespace

void vadd_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vadd(a, 1, b, 1, out, 1, L(n));
}

void vsub_f32(const float* a, const float* b, float* out, std::size_t n) {
    // vDSP_vsub computes C = B - A (note the order). We swap so the wrapper
    // reads naturally as out = a - b.
    vDSP_vsub(b, 1, a, 1, out, 1, L(n));
}

void vmul_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vmul(a, 1, b, 1, out, 1, L(n));
}

void vdiv_f32(const float* a, const float* b, float* out, std::size_t n) {
    // vDSP_vdiv: C = A / B (where A is the second arg; see vDSP docs).
    vDSP_vdiv(b, 1, a, 1, out, 1, L(n));
}

void vadd_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vaddD(a, 1, b, 1, out, 1, L(n));
}

void vsub_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vsubD(b, 1, a, 1, out, 1, L(n));
}

void vmul_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vmulD(a, 1, b, 1, out, 1, L(n));
}

void vdiv_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vdivD(b, 1, a, 1, out, 1, L(n));
}

void vneg_f32(const float* in, float* out, std::size_t n) {
    vDSP_vneg(in, 1, out, 1, L(n));
}

void vabs_f32(const float* in, float* out, std::size_t n) {
    vDSP_vabs(in, 1, out, 1, L(n));
}

void vsq_f32(const float* in, float* out, std::size_t n) {
    vDSP_vsq(in, 1, out, 1, L(n));
}

void vneg_f64(const double* in, double* out, std::size_t n) {
    vDSP_vnegD(in, 1, out, 1, L(n));
}

void vabs_f64(const double* in, double* out, std::size_t n) {
    vDSP_vabsD(in, 1, out, 1, L(n));
}

void vsq_f64(const double* in, double* out, std::size_t n) {
    vDSP_vsqD(in, 1, out, 1, L(n));
}

void vsadd_f32(const float* in, float scalar, float* out, std::size_t n) {
    vDSP_vsadd(in, 1, &scalar, out, 1, L(n));
}

void vsmul_f32(const float* in, float scalar, float* out, std::size_t n) {
    vDSP_vsmul(in, 1, &scalar, out, 1, L(n));
}

void vsadd_f64(const double* in, double scalar, double* out, std::size_t n) {
    vDSP_vsaddD(in, 1, &scalar, out, 1, L(n));
}

void vsmul_f64(const double* in, double scalar, double* out, std::size_t n) {
    vDSP_vsmulD(in, 1, &scalar, out, 1, L(n));
}

void vrelu_f32(const float* in, float* out, std::size_t n) {
    float zero = 0.0f;
    vDSP_vthres(in, 1, &zero, out, 1, L(n));
}

void vrelu_f64(const double* in, double* out, std::size_t n) {
    double zero = 0.0;
    vDSP_vthresD(in, 1, &zero, out, 1, L(n));
}

void vmax_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vmax(a, 1, b, 1, out, 1, L(n));
}

void vmin_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vmin(a, 1, b, 1, out, 1, L(n));
}

void vmax_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vmaxD(a, 1, b, 1, out, 1, L(n));
}

void vmin_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vminD(a, 1, b, 1, out, 1, L(n));
}

// vDSP doesn't ship a comparison-mask kernel for two vectors. Scalar loops
// below are simple and correct; if profiling shows hot spots we can replace
// with NEON intrinsics.
void vge_mask_f32(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
}

void vle_mask_f32(const float* a, const float* b, float* out, std::size_t n) {
    // Strict-less so ties go only to the >= side — matches PyTorch min/max
    // backward (no double-counting at equal values).
    for (std::size_t i = 0; i < n; ++i) out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

void vge_mask_f64(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = (a[i] >= b[i]) ? 1.0 : 0.0;
}

void vle_mask_f64(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = (a[i] < b[i]) ? 1.0 : 0.0;
}

void vadd_i32(const std::int32_t* a, const std::int32_t* b,
              std::int32_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void vadd_i64(const std::int64_t* a, const std::int64_t* b,
              std::int64_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

}  // namespace lucid::backend::cpu
