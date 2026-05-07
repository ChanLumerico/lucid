// lucid/_C/ops/fft/Fftn.h
//
// N-dimensional discrete Fourier transform (complex input → complex output).
//
// Forward:  out = mlx::core::fft::fftn(a, n, axes)
// Backward: handled at the Python layer (lucid.autograd.Function in lucid.fft)
//           — fft_backward(g) = ifft(g) with shape adjustments.
//
// The output is always complex64.  Real input is silently promoted by MLX.
// The output shape replaces in_shape[axes[i]] with n[i] when `n` is supplied;
// when `n` is empty, the input shape is preserved.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// `n` is the per-axis transform length (empty → use input sizes).
// `axes` is the list of dims to transform (empty → all dims).
// Negative axes are normalised to non-negative form by the implementation.
LUCID_API TensorImplPtr fftn_op(const TensorImplPtr& a,
                                const std::vector<std::int64_t>& n,
                                const std::vector<int>& axes);

}  // namespace lucid
