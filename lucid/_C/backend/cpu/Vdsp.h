// lucid/_C/backend/cpu/Vdsp.h
//
// Thin wrappers around Apple Accelerate vDSP vector arithmetic primitives used
// by the CPU backend.  Each function dispatches to the corresponding vDSP_*
// intrinsic (stride-1 paths) so the compiler and hardware can use NEON SIMD.
// "f32" variants wrap single-precision vDSP routines; "f64" wraps their "D"
// double-precision counterparts.  Integer add variants (i32, i64) have no vDSP
// equivalent and fall back to scalar loops.

#pragma once

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

// Elementwise vector addition: out[i] = a[i] + b[i].
LUCID_INTERNAL void vadd_f32(const float* a, const float* b, float* out, std::size_t n);
// Elementwise vector subtraction: out[i] = a[i] - b[i].
LUCID_INTERNAL void vsub_f32(const float* a, const float* b, float* out, std::size_t n);
// Elementwise vector multiplication: out[i] = a[i] * b[i].
LUCID_INTERNAL void vmul_f32(const float* a, const float* b, float* out, std::size_t n);
// Elementwise vector division: out[i] = a[i] / b[i].
LUCID_INTERNAL void vdiv_f32(const float* a, const float* b, float* out, std::size_t n);

// Double-precision elementwise add/sub/mul/div.
LUCID_INTERNAL void vadd_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vsub_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vmul_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vdiv_f64(const double* a, const double* b, double* out, std::size_t n);

// Elementwise unary negation: out[i] = -in[i].
LUCID_INTERNAL void vneg_f32(const float* in, float* out, std::size_t n);
// Elementwise absolute value: out[i] = |in[i]|.
LUCID_INTERNAL void vabs_f32(const float* in, float* out, std::size_t n);
// Elementwise square: out[i] = in[i]^2.  Uses vDSP_vsq (fused, no sqrt).
LUCID_INTERNAL void vsq_f32(const float* in, float* out, std::size_t n);

// Double-precision unary negation, absolute value, and square.
LUCID_INTERNAL void vneg_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vabs_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vsq_f64(const double* in, double* out, std::size_t n);

// Scalar-add broadcast: out[i] = in[i] + scalar.  Uses vDSP_vsadd.
LUCID_INTERNAL void vsadd_f32(const float* in, float scalar, float* out, std::size_t n);
// Scalar-multiply broadcast: out[i] = in[i] * scalar.  Uses vDSP_vsmul.
LUCID_INTERNAL void vsmul_f32(const float* in, float scalar, float* out, std::size_t n);
// Double-precision scalar add and scalar multiply.
LUCID_INTERNAL void vsadd_f64(const double* in, double scalar, double* out, std::size_t n);
LUCID_INTERNAL void vsmul_f64(const double* in, double scalar, double* out, std::size_t n);

// ReLU clamp: out[i] = max(0, in[i]).  Uses vDSP_vthres (threshold to zero).
LUCID_INTERNAL void vrelu_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vrelu_f64(const double* in, double* out, std::size_t n);

// Elementwise maximum: out[i] = max(a[i], b[i]).  Uses vDSP_vmax.
LUCID_INTERNAL void vmax_f32(const float* a, const float* b, float* out, std::size_t n);
// Elementwise minimum: out[i] = min(a[i], b[i]).  Uses vDSP_vmin.
LUCID_INTERNAL void vmin_f32(const float* a, const float* b, float* out, std::size_t n);
// Double-precision elementwise maximum and minimum.
LUCID_INTERNAL void vmax_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vmin_f64(const double* a, const double* b, double* out, std::size_t n);

// Boolean mask: out[i] = (a[i] >= b[i]) ? 1.0 : 0.0.  Scalar loop (no vDSP equivalent).
LUCID_INTERNAL void vge_mask_f32(const float* a, const float* b, float* out, std::size_t n);
// Boolean mask: out[i] = (a[i] < b[i]) ? 1.0 : 0.0.
LUCID_INTERNAL void vle_mask_f32(const float* a, const float* b, float* out, std::size_t n);
// Double-precision versions of the above two masks.
LUCID_INTERNAL void vge_mask_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vle_mask_f64(const double* a, const double* b, double* out, std::size_t n);

// Returns the sum of n elements.  Uses vDSP_sve (single-pass, numerically stable).
LUCID_INTERNAL float vsum_f32(const float* in, std::size_t n);
LUCID_INTERNAL double vsum_f64(const double* in, std::size_t n);
// Returns the arithmetic mean of n elements.  Uses vDSP_meanv.
LUCID_INTERNAL float vmean_f32(const float* in, std::size_t n);
LUCID_INTERNAL double vmean_f64(const double* in, std::size_t n);
// Returns the maximum value across n elements.  Uses vDSP_maxv.
LUCID_INTERNAL float vmaxval_f32(const float* in, std::size_t n);
LUCID_INTERNAL double vmaxval_f64(const double* in, std::size_t n);
// Returns the dot product of two n-element vectors.  Uses vDSP_dotpr.
LUCID_INTERNAL float vdotpr_f32(const float* a, const float* b, std::size_t n);
LUCID_INTERNAL double vdotpr_f64(const double* a, const double* b, std::size_t n);

// Fused multiply-add: out[i] = a[i]*b[i] + c[i].  Uses vDSP_vma.
LUCID_INTERNAL void
vmadd_f32(const float* a, const float* b, const float* c, float* out, std::size_t n);
LUCID_INTERNAL void
vmadd_f64(const double* a, const double* b, const double* c, double* out, std::size_t n);

// Integer addition for I32 and I64 element types.  No vDSP equivalent; uses
// scalar loops.  These are only exercised by integer-tensor add paths.
LUCID_INTERNAL void
vadd_i32(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n);
LUCID_INTERNAL void
vadd_i64(const std::int64_t* a, const std::int64_t* b, std::int64_t* out, std::size_t n);

}  // namespace lucid::backend::cpu
