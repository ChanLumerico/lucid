// lucid/_C/ops/fft/Fftn.h
//
// N-dimensional discrete Fourier transform (complex input to complex output).
//
// Forward dispatches to ``mlx::core::fft::fftn`` on the MLX CPU stream
// (Apple Silicon has no Metal FFT kernel yet; MLX transitively reaches
// Accelerate vDSP under the hood).  Backward is wired at the Python layer
// in ``lucid.fft`` as an ``autograd.Function`` whose VJP calls ``ifftn``
// on the incoming gradient with the appropriate shape adjustments.
//
// The output is always complex64.  Real input (F16/F32) is silently
// promoted to complex by MLX.  The output shape is the input shape with
// each ``axes[i]`` replaced by ``n[i]`` when ``n`` is supplied; when
// ``n`` is empty, the input shape is preserved verbatim.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// N-dimensional complex-to-complex DFT (forward).
//
// Computes the discrete Fourier transform of $x$ along the requested
// ``axes``.  This is the entry point invoked by ``lucid.fft.fftn`` and
// is consumed by Python-side autograd machinery; this function itself
// performs no gradient wiring.
//
// Math
// ----
// For a $d$-dimensional transform over axes $a_1, \dots, a_d$ of sizes
// $N_1, \dots, N_d$,
//
// $$
//   X[k_1, \dots, k_d] = \sum_{n_1=0}^{N_1-1} \cdots \sum_{n_d=0}^{N_d-1}
//     x[n_1, \dots, n_d] \,
//     \exp\!\left( -2\pi i \sum_{j=1}^{d} \frac{k_j n_j}{N_j} \right) .
// $$
//
// The ``"backward"`` normalisation convention is used (no scaling on
// forward; division by $\prod_j N_j$ on inverse).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  Real (F16/F32) inputs are accepted and promoted
//     to C64; complex (C64) inputs are passed through.
// n : vector<int64_t>
//     Per-axis transform length.  When empty, the corresponding input
//     dimensions are used.  When non-empty, ``len(n) == len(axes)``
//     must hold; each input axis is cropped or zero-padded to ``n[i]``.
// axes : vector<int>
//     Axes to transform.  Negative axes are normalised against the
//     input rank.  Empty means "transform all axes".
//
// Returns
// -------
// TensorImplPtr
//     C64 tensor with shape equal to ``a.shape()`` except that each
//     ``axes[i]`` is replaced by ``n[i]`` (or unchanged when ``n`` is
//     empty).
//
// Shape
// -----
// ``out.shape[axes[i]] == n[i]`` when ``n`` is supplied; otherwise
// ``out.shape == a.shape``.  All other axes retain their input size.
//
// Notes
// -----
// FFT bypasses the IBackend dispatcher and calls MLX directly from
// both CPU and GPU device paths (see ``_Detail.h``).  On Device::CPU
// the result is downloaded to host memory after the transform.
//
// Raises
// ------
// LucidError
//     If ``a`` is null, rank-0, has an unsupported dtype, or if
//     ``len(n) != len(axes)`` when ``n`` is non-empty.
//
// References
// ----------
// Cooley & Tukey, "An Algorithm for the Machine Calculation of Complex
// Fourier Series" (Math. Comp., 1965).
//
// See Also
// --------
// ifftn_op : Inverse N-D DFT (backward of ``fftn``).
// rfftn_op : Real-input N-D DFT.
LUCID_API TensorImplPtr fftn_op(const TensorImplPtr& a,
                                const std::vector<std::int64_t>& n,
                                const std::vector<int>& axes);

}  // namespace lucid
