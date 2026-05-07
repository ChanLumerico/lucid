// lucid/_C/ops/fft/Irfftn.h
//
// N-dimensional inverse real DFT (complex input → real output).
// Forward: mlx::core::fft::irfftn(a, n, axes).
// Output dtype: F32.  Last specified axis becomes n[-1] (default 2*(in[ax]-1)).
// Backward (Python layer): rfftn(g) with shape adjustments.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr irfftn_op(const TensorImplPtr& a,
                                  const std::vector<std::int64_t>& n,
                                  const std::vector<int>& axes);

}  // namespace lucid
