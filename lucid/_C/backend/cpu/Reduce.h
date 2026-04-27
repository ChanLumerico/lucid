#pragma once

// =====================================================================
// Lucid C++ engine — axis-aware reduction kernels.
// =====================================================================
//
// Each kernel reduces a single contiguous axis. Multi-axis reductions are
// implemented at the op level by sequentially reducing one axis at a time
// (descending order so lower-indexed axes' positions stay stable between
// passes).
//
// Convention: input is logically [outer, reduce_dim, inner]:
//   outer      = product of dims before the reduce axis
//   reduce_dim = size of the axis being reduced
//   inner      = product of dims after the reduce axis
// Output shape is [outer, inner] (axis squeezed). Use keepdims=true
// reshape at the op level if needed.
//
// Layer: backend/cpu/.

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

LUCID_INTERNAL void sum_axis_f32(const float* in, float* out,
                                 std::size_t outer, std::size_t reduce_dim,
                                 std::size_t inner);
LUCID_INTERNAL void sum_axis_f64(const double* in, double* out,
                                 std::size_t outer, std::size_t reduce_dim,
                                 std::size_t inner);

LUCID_INTERNAL void max_axis_f32(const float* in, float* out,
                                 std::size_t outer, std::size_t reduce_dim,
                                 std::size_t inner);
LUCID_INTERNAL void max_axis_f64(const double* in, double* out,
                                 std::size_t outer, std::size_t reduce_dim,
                                 std::size_t inner);

LUCID_INTERNAL void min_axis_f32(const float* in, float* out,
                                 std::size_t outer, std::size_t reduce_dim,
                                 std::size_t inner);
LUCID_INTERNAL void min_axis_f64(const double* in, double* out,
                                 std::size_t outer, std::size_t reduce_dim,
                                 std::size_t inner);

LUCID_INTERNAL void prod_axis_f32(const float* in, float* out,
                                  std::size_t outer, std::size_t reduce_dim,
                                  std::size_t inner);
LUCID_INTERNAL void prod_axis_f64(const double* in, double* out,
                                  std::size_t outer, std::size_t reduce_dim,
                                  std::size_t inner);

}  // namespace lucid::backend::cpu
