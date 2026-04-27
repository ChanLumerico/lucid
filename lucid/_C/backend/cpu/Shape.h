#pragma once

// =====================================================================
// Lucid C++ engine — shape-op kernels (permute / flip).
// =====================================================================
//
// Materialized-copy implementations of axis permutations. Phase 3.4 v1
// always copies into a fresh contiguous buffer, keeping the engine's
// contiguous-input invariant trivial. Future zero-copy view ops would
// flip the contiguous guard inside compute paths instead.
//
// Layer: backend/cpu/.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../api.h"

namespace lucid::backend::cpu {

/// Copy `in` (laid out contiguously in `in_shape`) into `out` (laid out
/// contiguously in the permuted shape). `perm` is a permutation of
/// 0..ndim-1; `out_shape[i] = in_shape[perm[i]]`.
LUCID_INTERNAL void permute_copy_f32(const float* in, float* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);
LUCID_INTERNAL void permute_copy_f64(const double* in, double* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);
LUCID_INTERNAL void permute_copy_i32(const std::int32_t* in, std::int32_t* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);
LUCID_INTERNAL void permute_copy_i64(const std::int64_t* in, std::int64_t* out,
                                     const std::vector<std::int64_t>& in_shape,
                                     const std::vector<int>& perm);

}  // namespace lucid::backend::cpu
