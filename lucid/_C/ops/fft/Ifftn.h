// lucid/_C/ops/fft/Ifftn.h
//
// N-dimensional inverse DFT (complex input → complex output).
// Forward: mlx::core::fft::ifftn(a, n, axes).  Output is C64.
// Backward (Python layer): ifft_backward(g) = fft(g) / N (handled by lucid.fft).

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr ifftn_op(const TensorImplPtr& a,
                                 const std::vector<std::int64_t>& n,
                                 const std::vector<int>& axes);

}  // namespace lucid
