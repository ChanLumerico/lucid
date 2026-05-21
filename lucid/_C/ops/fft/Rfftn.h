// lucid/_C/ops/fft/Rfftn.h
//
// N-dimensional real-input discrete Fourier transform.
//
// Real inputs produce Hermitian-symmetric spectra, so the redundant
// negative-frequency half is dropped: the last transformed axis is
// stored at length $n_{\text{last}}/2 + 1$.  Forward dispatches to
// ``mlx::core::fft::rfftn`` on the MLX CPU stream.  Backward is wired
// at the Python layer in ``lucid.fft`` as an ``autograd.Function``
// whose VJP calls ``irfftn`` on the gradient with the original real
// input shape.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// N-dimensional real-to-complex DFT.
//
// Computes the DFT of a real signal and stores only the non-redundant
// half along the last transformed axis (the Hermitian half).  The
// remaining frequencies can be reconstructed by complex conjugation;
// this storage convention matches ``numpy.fft.rfftn``.
//
// Math
// ----
// For real input $x$,
//
// $$
//   X[k_1, \dots, k_d] = \sum_{n_1=0}^{N_1-1} \cdots \sum_{n_d=0}^{N_d-1}
//     x[n_1, \dots, n_d] \,
//     \exp\!\left( -2\pi i \sum_{j=1}^{d} \frac{k_j n_j}{N_j} \right) ,
// $$
//
// stored over $k_d \in \{0, 1, \dots, N_d/2\}$ only (the last axis is
// trimmed by Hermitian symmetry; all other axes are stored in full).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Real input tensor (F16/F32).
// n : vector<int64_t>
//     Per-axis transform length.  Empty means "use input sizes".
//     When supplied, ``len(n) == len(axes)`` is required, and the last
//     transformed axis is stored at ``n.back() / 2 + 1``.
// axes : vector<int>
//     Axes to transform.  Must be non-empty (rfftn requires at least
//     one axis).  Negative axes are normalised against the input rank.
//
// Returns
// -------
// TensorImplPtr
//     C64 tensor.  Shape matches the input except for the last entry
//     of ``axes``, which becomes ``(n_last) / 2 + 1`` where ``n_last``
//     is ``n.back()`` (or ``a.shape[axes.back()]`` when ``n`` is empty).
//
// Shape
// -----
// Let ``L = axes.back()``.  Then ``out.shape[L] == n.back() / 2 + 1``
// when ``n`` is supplied, else ``a.shape[L] / 2 + 1``.  All other
// transformed axes follow ``n`` (or the input shape).
//
// Notes
// -----
// FFT bypasses IBackend dispatch and calls MLX directly (see
// ``_Detail.h``).  The Python wrapper must remember the original last
// axis size to disambiguate the inverse.
//
// Raises
// ------
// LucidError
//     If ``a`` is null, has a non-real dtype, ``axes`` is empty, or
//     ``len(n) != len(axes)`` when ``n`` is non-empty.
//
// See Also
// --------
// irfftn_op : Inverse real DFT.
// fftn_op : Full complex-to-complex DFT.
LUCID_API TensorImplPtr rfftn_op(const TensorImplPtr& a,
                                 const std::vector<std::int64_t>& n,
                                 const std::vector<int>& axes);

}  // namespace lucid
