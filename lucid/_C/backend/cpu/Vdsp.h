#pragma once

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

LUCID_INTERNAL void vadd_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vsub_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vmul_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vdiv_f32(const float* a, const float* b, float* out, std::size_t n);

LUCID_INTERNAL void vadd_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vsub_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vmul_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vdiv_f64(const double* a, const double* b, double* out, std::size_t n);

LUCID_INTERNAL void vneg_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vabs_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vsq_f32(const float* in, float* out, std::size_t n);

LUCID_INTERNAL void vneg_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vabs_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vsq_f64(const double* in, double* out, std::size_t n);

LUCID_INTERNAL void vsadd_f32(const float* in, float scalar, float* out, std::size_t n);
LUCID_INTERNAL void vsmul_f32(const float* in, float scalar, float* out, std::size_t n);
LUCID_INTERNAL void vsadd_f64(const double* in, double scalar, double* out, std::size_t n);
LUCID_INTERNAL void vsmul_f64(const double* in, double scalar, double* out, std::size_t n);

LUCID_INTERNAL void vrelu_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vrelu_f64(const double* in, double* out, std::size_t n);

LUCID_INTERNAL void vmax_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vmin_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vmax_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vmin_f64(const double* a, const double* b, double* out, std::size_t n);

LUCID_INTERNAL void vge_mask_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vle_mask_f32(const float* a, const float* b, float* out, std::size_t n);
LUCID_INTERNAL void vge_mask_f64(const double* a, const double* b, double* out, std::size_t n);
LUCID_INTERNAL void vle_mask_f64(const double* a, const double* b, double* out, std::size_t n);

LUCID_INTERNAL float vsum_f32(const float* in, std::size_t n);
LUCID_INTERNAL double vsum_f64(const double* in, std::size_t n);
LUCID_INTERNAL float vmean_f32(const float* in, std::size_t n);
LUCID_INTERNAL double vmean_f64(const double* in, std::size_t n);
LUCID_INTERNAL float vmaxval_f32(const float* in, std::size_t n);
LUCID_INTERNAL double vmaxval_f64(const double* in, std::size_t n);
LUCID_INTERNAL float vdotpr_f32(const float* a, const float* b, std::size_t n);
LUCID_INTERNAL double vdotpr_f64(const double* a, const double* b, std::size_t n);

LUCID_INTERNAL void
vmadd_f32(const float* a, const float* b, const float* c, float* out, std::size_t n);
LUCID_INTERNAL void
vmadd_f64(const double* a, const double* b, const double* c, double* out, std::size_t n);

LUCID_INTERNAL void
vadd_i32(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n);
LUCID_INTERNAL void
vadd_i64(const std::int64_t* a, const std::int64_t* b, std::int64_t* out, std::size_t n);

}  // namespace lucid::backend::cpu
