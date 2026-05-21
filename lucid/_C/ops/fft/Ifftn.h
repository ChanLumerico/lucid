// lucid/_C/ops/fft/Ifftn.h
//
// N-dimensional inverse discrete Fourier transform (complex to complex).
//
// Forward dispatches to ``mlx::core::fft::ifftn`` on the MLX CPU stream.
// Backward is wired at the Python layer in ``lucid.fft`` as an
// ``autograd.Function`` whose VJP calls ``fftn`` on the gradient with
// the appropriate shape adjustments.
//
// Output is always C64; the input may be C64, F32, or F16 (real inputs
// are promoted by MLX).

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// N-dimensional inverse complex-to-complex DFT (forward of the inverse).
//
// Inverts an N-D DFT.  Uses the ``"backward"`` normalisation
// convention: forward has no scaling, this op divides by $\prod_j N_j$
// where $N_j$ is the transform length along each axis.
//
// Math
// ----
// $$
//   x[n_1, \dots, n_d] = \frac{1}{\prod_j N_j}
//     \sum_{k_1=0}^{N_1-1} \cdots \sum_{k_d=0}^{N_d-1}
//     X[k_1, \dots, k_d] \,
//     \exp\!\left( +2\pi i \sum_{j=1}^{d} \frac{k_j n_j}{N_j} \right) .
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  C64 expected; F16/F32 are accepted and promoted.
// n : vector<int64_t>
//     Per-axis transform length.  Empty means "use input sizes".
//     When supplied, ``len(n) == len(axes)`` is required.
// axes : vector<int>
//     Axes to transform.  Negative values are normalised against the
//     input rank.  Empty means "transform all axes".
//
// Returns
// -------
// TensorImplPtr
//     C64 tensor of the same shape as ``a`` (with ``axes[i]`` replaced
//     by ``n[i]`` when ``n`` is supplied).
//
// Notes
// -----
// FFT bypasses IBackend dispatch and calls MLX directly from both CPU
// and GPU device paths (see ``_Detail.h``).
//
// Raises
// ------
// LucidError
//     If ``a`` is null, rank-0, has an unsupported dtype, or if
//     ``len(n) != len(axes)`` when ``n`` is non-empty.
//
// See Also
// --------
// fftn_op : Forward N-D DFT.
// irfftn_op : Inverse of the real-input DFT.
LUCID_API TensorImplPtr ifftn_op(const TensorImplPtr& a,
                                 const std::vector<std::int64_t>& n,
                                 const std::vector<int>& axes);

}  // namespace lucid
