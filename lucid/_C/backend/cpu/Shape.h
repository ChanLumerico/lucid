// lucid/_C/backend/cpu/Shape.h
//
// CPU shape-transformation helper: permute_copy performs an N-D transpose by
// copying elements in the permuted order into a fresh densely-packed buffer.
// This is used by CpuBackend::permute_cpu() and by the GPU backend's tensordot
// data-layout preparation path.
//
// The permutation perm[d] specifies which input axis maps to output axis d,
// following NumPy conventions (e.g. perm = {2, 0, 1} maps (H, W, C) → (C, H, W)).
// Output strides are computed from the output shape in C (row-major) order.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../api.h"

namespace lucid::backend::cpu {

// Permutes in of shape in_shape according to perm and writes the result to out.
// out must be pre-allocated with numel(in_shape) * sizeof(T) bytes.
LUCID_INTERNAL void permute_copy_f32(const float* in,
                                     float* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);
LUCID_INTERNAL void permute_copy_f64(const double* in,
                                     double* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);
LUCID_INTERNAL void permute_copy_i32(const std::int32_t* in,
                                     std::int32_t* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);
LUCID_INTERNAL void permute_copy_i64(const std::int64_t* in,
                                     std::int64_t* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);

}  // namespace lucid::backend::cpu
