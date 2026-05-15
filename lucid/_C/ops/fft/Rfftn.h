// lucid/_C/ops/fft/Rfftn.h
//
// N-dimensional real-input DFT (real input → complex output).
// Forward: mlx::core::fft::rfftn(a, n, axes).
// Output dtype: C64.  Output shape: last specified axis becomes n[-1] // 2 + 1.
// Backward (Python layer): irfftn(g, original_shape).

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr rfftn_op(const TensorImplPtr& a,
                                 const std::vector<std::int64_t>& n,
                                 const std::vector<int>& axes);

}  // namespace lucid
