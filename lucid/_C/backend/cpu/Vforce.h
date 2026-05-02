#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

LUCID_INTERNAL void vexp_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vlog_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vsqrt_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vtanh_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vsin_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vcos_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vtan_f32(const float* in, float* out, std::size_t n);

LUCID_INTERNAL void vexp_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vlog_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vsqrt_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vtanh_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vsin_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vcos_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vtan_f64(const double* in, double* out, std::size_t n);

LUCID_INTERNAL void vpow_f32(const float* base, const float* expo, float* out, std::size_t n);
LUCID_INTERNAL void vpow_f64(const double* base, const double* expo, double* out, std::size_t n);

LUCID_INTERNAL void vasin_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vacos_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vatan_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vasin_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vacos_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vatan_f64(const double* in, double* out, std::size_t n);

LUCID_INTERNAL void vsinh_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vcosh_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vsinh_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vcosh_f64(const double* in, double* out, std::size_t n);

LUCID_INTERNAL void vlog2_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vlog2_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vfabs_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vfabs_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vrec_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vrec_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vfloor_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vfloor_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vceil_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vceil_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vround_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vround_f64(const double* in, double* out, std::size_t n);

}  // namespace lucid::backend::cpu
