// lucid/_C/backend/cpu/Reduce.h
//
// Single-axis reduction primitives used by the CPU backend.  Each function
// reduces one axis of a tensor described by three extent values:
//   outer       — product of all dimensions before the reduce axis
//   reduce_dim  — size of the dimension being reduced
//   inner       — product of all dimensions after the reduce axis
//
// The input layout follows row-major order: element [o, r, i] lives at
// in[o * reduce_dim * inner + r * inner + i].  The output has shape
// [outer, inner] and is written row-major.
//
// sum_axis_f32 uses vDSP_sve for the inner == 1 (contiguous) case because
// vDSP provides a numerically stable single-pass summation that is faster
// than the scalar loop.  All other operations use the generic axis_reduce
// template defined in Reduce.cpp.

#pragma once

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

// Sums elements along the reduce dimension: out[o, i] = sum_r in[o, r, i].
// When inner == 1 the f32 path delegates to vDSP_sve for performance.
LUCID_INTERNAL void sum_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void sum_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Computes the maximum: out[o, i] = max_r in[o, r, i].  Identity is -inf.
LUCID_INTERNAL void max_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void max_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Computes the minimum: out[o, i] = min_r in[o, r, i].  Identity is +inf.
LUCID_INTERNAL void min_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void min_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Computes the product: out[o, i] = prod_r in[o, r, i].  Identity is 1.
LUCID_INTERNAL void prod_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void prod_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

}  // namespace lucid::backend::cpu
