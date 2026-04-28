#pragma once

// =====================================================================
// Lucid C++ engine — Apple vDSP wrappers (CPU element-wise math).
// =====================================================================
//
// Thin contiguous-stride wrappers around `<Accelerate/Accelerate.h>` vDSP_*
// routines. Each function takes `(in..., out, n)` with n elements; strides
// are 1 (we don't expose strided ops — strided ops compose by reshape).
//
// Naming: `<op>_<dtype>` — so `vadd_f32`, `vmul_f64`, `vthres_f32`, etc.
// dtype dispatch happens at the call site (op kernels do `switch (dtype)`).
//
// Layer: backend/cpu/. May include from core/ only.

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

// element-wise binary
LUCID_INTERNAL void vadd_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vsub_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vmul_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vdiv_f32(const float* a, const float* b, float* out, std::size_t n);

LUCID_INTERNAL void vadd_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vsub_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vmul_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vdiv_f64(const double* a, const double* b, double* out, std::size_t n);

// element-wise unary
LUCID_INTERNAL void vneg_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vabs_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vsq_f32(const float* in, float* out, std::size_t n);  // x^2

LUCID_INTERNAL void vneg_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vabs_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vsq_f64(const double* in, double* out, std::size_t n);

// scalar-vector
LUCID_INTERNAL void vsadd_f32(const float* in, float scalar, float* out, std::size_t n);
LUCID_INTERNAL void vsmul_f32(const float* in, float scalar, float* out, std::size_t n);
LUCID_INTERNAL void vsadd_f64(const double* in, double scalar, double* out, std::size_t n);
LUCID_INTERNAL void vsmul_f64(const double* in, double scalar, double* out, std::size_t n);

// activation: relu(x) = max(x, 0) via vDSP_vthres at threshold 0
LUCID_INTERNAL void vrelu_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vrelu_f64(const double* in, double* out, std::size_t n);

// element-wise max/min between two vectors (used by Maximum/Minimum forward)
LUCID_INTERNAL void vmax_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vmin_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vmax_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vmin_f64(const double* a, const double* b, double* out, std::size_t n);

// element-wise mask: out[i] = (a[i] OP b[i]) ? 1 : 0  (used by min/max backward).
// Tied case (a[i] == b[i]) goes to the `_ge` mask, mirroring PyTorch.
LUCID_INTERNAL void vge_mask_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vle_mask_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vge_mask_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vle_mask_f64(const double* a, const double* b, double* out, std::size_t n);

// Integer fallbacks (vDSP doesn't ship these natively; we use scalar loops
// — they exist so the dispatch table is complete in Phase 3.0. Phase 3.1+
// can replace with platform-specific accelerated paths if profiling demands.)
LUCID_INTERNAL void vadd_i32(const std::int32_t* a,
                             const std::int32_t* b,
                             std::int32_t* out,
                             std::size_t n);
LUCID_INTERNAL void vadd_i64(const std::int64_t* a,
                             const std::int64_t* b,
                             std::int64_t* out,
                             std::size_t n);

}  // namespace lucid::backend::cpu
