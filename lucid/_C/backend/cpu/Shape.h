#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../api.h"

namespace lucid::backend::cpu {

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
