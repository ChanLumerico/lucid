// lucid/_C/ops/utils/Histogram.h
//
// Declares histogram operations for 1-D, 2-D, and N-D data.  All three
// functions are non-differentiable; they return floating-point bin counts
// (and optionally densities) together with the bin edge positions.
//
// Storage note: histogram computation is always performed on CPU because
// counting requires random-access writes with data-dependent addressing that
// does not map well to GPU kernels.  If the input resides on the GPU it is
// first copied to CPU.  F64 outputs always remain on CPU (MLX does not
// support float64 buffers); other dtypes are transferred back to the
// requested device after counting.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute a 1-D histogram of values in ``a`` over uniform bins.
//
// Partitions the half-open interval $[lo, hi)$ into ``bins`` equal-width
// sub-intervals and counts the number of elements of ``a`` falling in each
// one.  When ``density`` is true the counts are normalised so that the
// histogram integrates to one over the support: counts are divided by
// ``(bin_width * N)`` where ``N`` is the total number of input elements.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape; flattened internally.
// bins : int64
//     Number of equal-width bins.  Must be positive.
// lo : double
//     Left edge of the lowest bin (inclusive).
// hi : double
//     Right edge of the highest bin (exclusive, except for the last bin
//     which is closed on the right).
// density : bool
//     If true, return a probability density; otherwise return raw counts.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``[counts, edges]`` where ``counts`` has shape ``(bins,)`` and
//     ``edges`` has shape ``(bins + 1,)``.  Both have dtype F64.
//
// Math
// ----
// $$counts_i = \#\{k : lo + i \Delta \le a_k < lo + (i+1)\Delta\}, \quad
// \Delta = (hi - lo) / bins$$
//
// Notes
// -----
// Non-differentiable: counts are piecewise-constant in the input.  Always
// executes on the CPU stream; GPU inputs are copied across.
//
// See Also
// --------
// histogram2d_op : Joint 2-D histogram of paired observations.
// histogramdd_op : Histogram of (N, D) row-major samples in D dimensions.
LUCID_API std::vector<TensorImplPtr>
histogram_op(const TensorImplPtr& a, std::int64_t bins, double lo, double hi, bool density);

// Compute a 2-D histogram from two equal-length 1-D tensors.
//
// Treats paired observations ``(a_k, b_k)`` as samples on an
// $\mathrm{bins\_a} \times \mathrm{bins\_b}$ grid of uniform rectangular
// cells and counts the number of pairs falling in each cell.  When
// ``density`` is true each cell value is divided by
// ``(step_a * step_b * N)``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     1-D tensor of x-coordinates.
// b : TensorImplPtr
//     1-D tensor of y-coordinates.  Must match ``a`` in length, dtype,
//     and device.
// bins_a : int64
//     Number of bins along the x-axis.
// bins_b : int64
//     Number of bins along the y-axis.
// lo_a, hi_a : double
//     Range of the x-axis.
// lo_b, hi_b : double
//     Range of the y-axis.
// density : bool
//     If true, normalise to a probability density.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``[counts, edges]`` where ``counts`` has shape ``(bins_a, bins_b)``
//     and ``edges`` has shape ``(bins_a + 1 + bins_b + 1,)`` — the x-edges
//     followed by the y-edges concatenated.  Both dtype F64.
//
// Notes
// -----
// Non-differentiable.  Executes on the CPU stream regardless of input
// device.
//
// See Also
// --------
// histogram_op : 1-D variant.
// histogramdd_op : General N-dimensional variant.
LUCID_API std::vector<TensorImplPtr> histogram2d_op(const TensorImplPtr& a,
                                                    const TensorImplPtr& b,
                                                    std::int64_t bins_a,
                                                    std::int64_t bins_b,
                                                    double lo_a,
                                                    double hi_a,
                                                    double lo_b,
                                                    double hi_b,
                                                    bool density);

// Compute an N-dimensional histogram from a 2-D sample matrix.
//
// Given an input tensor of shape ``(N, D)`` whose rows are D-dimensional
// observations, partitions each dimension into the corresponding number
// of bins and counts samples falling in each D-dimensional cell.  When
// ``density`` is true each cell is divided by ``(cell_volume * N)``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     2-D tensor of shape ``(N, D)``; row ``k`` is the ``k``-th sample.
// bins : vector<int64>
//     Length-D vector giving the bin count along each dimension.
// ranges : vector<pair<double, double>>
//     Length-D vector of ``(lo, hi)`` pairs, one range per dimension.
// density : bool
//     If true, normalise to a probability density.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``[counts, edges]`` where ``counts`` is a D-dimensional tensor
//     whose shape equals ``bins``, and ``edges`` is a 1-D tensor of
//     length ``sum(bins[d] + 1)`` containing the per-dimension edge
//     arrays concatenated in order.  Both dtype F64.
//
// Notes
// -----
// Non-differentiable.  Always executes on the CPU stream.
//
// See Also
// --------
// histogram_op, histogram2d_op : Lower-dimensional specialisations.
LUCID_API std::vector<TensorImplPtr> histogramdd_op(const TensorImplPtr& a,
                                                    std::vector<std::int64_t> bins,
                                                    std::vector<std::pair<double, double>> ranges,
                                                    bool density);

}  // namespace lucid
