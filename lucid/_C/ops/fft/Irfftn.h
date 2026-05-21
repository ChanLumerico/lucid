// lucid/_C/ops/fft/Irfftn.h
//
// Inverse of ``rfftn``: takes a Hermitian-half complex spectrum and
// reconstructs the real signal.  Forward dispatches to
// ``mlx::core::fft::irfftn`` on the MLX CPU stream.  Backward is wired
// at the Python layer in ``lucid.fft`` as an ``autograd.Function``
// whose VJP calls ``rfftn`` on the gradient.
//
// The output is always F32.  The caller must supply (via ``n``) the
// original real-input length along the last transformed axis to
// disambiguate even vs. odd reconstructions; when ``n`` is empty, the
// default ``2 * (in_shape[axes.back()] - 1)`` is used (even-length
// assumption).

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// N-dimensional inverse real DFT (complex Hermitian-half to real).
//
// Reconstructs the real signal from a Hermitian-half spectrum produced
// by ``rfftn``.  The last transformed axis is expanded from the stored
// length $n_{\text{last}}/2 + 1$ back to $n_{\text{last}}$.  Uses the
// ``"backward"`` normalisation convention (divides by $\prod_j N_j$).
//
// Math
// ----
// $$
//   x[n_1, \dots, n_d] = \frac{1}{\prod_j N_j}
//     \sum_{k_1=0}^{N_1-1} \cdots \sum_{k_d=0}^{N_d-1}
//     X[k_1, \dots, k_d] \,
//     \exp\!\left( +2\pi i \sum_{j=1}^{d} \frac{k_j n_j}{N_j} \right) ,
// $$
//
// reconstructed assuming $X$ is Hermitian-symmetric along the last
// transformed axis ($X[\dots, N_d - k_d] = \overline{X[\dots, k_d]}$).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Complex input (C64), shaped as the output of ``rfftn``.
// n : vector<int64_t>
//     Per-axis transform length.  The last entry specifies the original
//     real-input length along ``axes.back()``; when ``n`` is empty,
//     the default ``2 * (a.shape[axes.back()] - 1)`` is used.
// axes : vector<int>
//     Axes to transform.  Must be non-empty.  Negative values are
//     normalised against the input rank.
//
// Returns
// -------
// TensorImplPtr
//     F32 tensor.  Shape follows ``a`` except that ``axes[i]`` is
//     replaced by ``n[i]`` (with the default applied to the last axis
//     when ``n`` is empty).
//
// Shape
// -----
// ``out.shape[axes[i]] == n[i]`` when ``n`` is supplied; otherwise
// ``out.shape[axes.back()] == 2 * (a.shape[axes.back()] - 1)`` and
// other transformed axes are unchanged.
//
// Notes
// -----
// FFT bypasses IBackend dispatch and calls MLX directly (see
// ``_Detail.h``).  When the original real length was odd, ``n.back()``
// must be supplied explicitly — the default formula assumes even length.
//
// Raises
// ------
// LucidError
//     If ``a`` is null, not C64, ``axes`` is empty, or
//     ``len(n) != len(axes)`` when ``n`` is non-empty.
//
// See Also
// --------
// rfftn_op : Forward real-input DFT.
// ifftn_op : Inverse of full complex-to-complex DFT.
LUCID_API TensorImplPtr irfftn_op(const TensorImplPtr& a,
                                  const std::vector<std::int64_t>& n,
                                  const std::vector<int>& axes);

}  // namespace lucid
