// lucid/_C/backend/cpu/Vforce.h
//
// Thin wrappers around Apple Accelerate vForce transcendental math routines.
// All functions apply the corresponding vv* vector math function element-wise
// to an array of n values.  vForce functions are vectorized for Apple Silicon
// NEON and are significantly faster than calling std:: math functions in a loop.
//
// Naming convention: v<op>_f32 wraps the vv*f (float) variant; v<op>_f64
// wraps the vv* (double) variant.  The element count n is passed as an int*
// to match the vForce API signature.

#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// Exponential: out[i] = e^in[i].  Uses vvexpf / vvexp.
LUCID_INTERNAL void vexp_f32(const float* in, float* out, std::size_t n);
// Natural log: out[i] = ln(in[i]).  Uses vvlogf / vvlog.
LUCID_INTERNAL void vlog_f32(const float* in, float* out, std::size_t n);
// Square root: out[i] = sqrt(in[i]).  Uses vvsqrtf / vvsqrt.
LUCID_INTERNAL void vsqrt_f32(const float* in, float* out, std::size_t n);
// Hyperbolic tangent: out[i] = tanh(in[i]).  Uses vvtanhf / vvtanh.
LUCID_INTERNAL void vtanh_f32(const float* in, float* out, std::size_t n);
// Trigonometric sine: out[i] = sin(in[i]).  Uses vvsinf / vvsin.
LUCID_INTERNAL void vsin_f32(const float* in, float* out, std::size_t n);
// Trigonometric cosine: out[i] = cos(in[i]).  Uses vvcosf / vvcos.
LUCID_INTERNAL void vcos_f32(const float* in, float* out, std::size_t n);
// Trigonometric tangent: out[i] = tan(in[i]).  Uses vvtanf / vvtan.
LUCID_INTERNAL void vtan_f32(const float* in, float* out, std::size_t n);

// Double-precision variants for exp, log, sqrt, tanh, sin, cos, tan.
LUCID_INTERNAL void vexp_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vlog_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vsqrt_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vtanh_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vsin_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vcos_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vtan_f64(const double* in, double* out, std::size_t n);

// Elementwise power: out[i] = base[i]^expo[i].
// Uses vvpowf / vvpow; note that vvpow* has (out, expo, base, &n) argument
// order (exponent before base), which the wrappers reorder for clarity.
LUCID_INTERNAL void vpow_f32(const float* base, const float* expo, float* out, std::size_t n);
LUCID_INTERNAL void vpow_f64(const double* base, const double* expo, double* out, std::size_t n);

// Inverse trigonometric functions (asin, acos, atan).
// Uses vvasinf/vvasin, vvacosf/vvacos, vvatanf/vvatan.
LUCID_INTERNAL void vasin_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vacos_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vatan_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vasin_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vacos_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vatan_f64(const double* in, double* out, std::size_t n);

// Hyperbolic sinh and cosh.  Uses vvsinhf/vvsinh, vvcoshf/vvcosh.
LUCID_INTERNAL void vsinh_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vcosh_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vsinh_f64(const double* in, double* out, std::size_t n);
LUCID_INTERNAL void vcosh_f64(const double* in, double* out, std::size_t n);

// Base-2 logarithm: out[i] = log2(in[i]).  Uses vvlog2f / vvlog2.
LUCID_INTERNAL void vlog2_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vlog2_f64(const double* in, double* out, std::size_t n);
// Error function: out[i] = erf(in[i]).  Uses vverff / vverf.
LUCID_INTERNAL void verf_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void verf_f64(const double* in, double* out, std::size_t n);
// Floating-point absolute value via vvfabsf / vvfabs.
LUCID_INTERNAL void vfabs_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vfabs_f64(const double* in, double* out, std::size_t n);
// Reciprocal: out[i] = 1/in[i].  Uses vvrecf / vvrec.
LUCID_INTERNAL void vrec_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vrec_f64(const double* in, double* out, std::size_t n);
// Floor: out[i] = floor(in[i]).  Uses vvfloorf / vvfloor.
LUCID_INTERNAL void vfloor_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vfloor_f64(const double* in, double* out, std::size_t n);
// Ceiling: out[i] = ceil(in[i]).  Uses vvceilf / vvceil.
LUCID_INTERNAL void vceil_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vceil_f64(const double* in, double* out, std::size_t n);
// Round-to-nearest: out[i] = round(in[i]).  Uses vvnintf / vvnint (round to
// nearest integer, ties to even).
LUCID_INTERNAL void vround_f32(const float* in, float* out, std::size_t n);
LUCID_INTERNAL void vround_f64(const double* in, double* out, std::size_t n);

}  // namespace lucid::backend::cpu
