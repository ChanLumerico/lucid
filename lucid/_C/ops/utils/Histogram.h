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

// Compute a 1-D histogram of `a` over `bins` uniform bins in [lo, hi).
//
// Returns {counts, edges} where:
//   counts — 1-D tensor of shape (bins,), dtype F64.
//   edges  — 1-D tensor of shape (bins+1,), dtype F64, with uniformly spaced
//             bin boundaries.  The last edge is exactly `hi`.
//
// If `density` is true, counts are divided by (bin_width * total_elements)
// so that the histogram integrates to 1.
LUCID_API std::vector<TensorImplPtr>
histogram_op(const TensorImplPtr& a, std::int64_t bins, double lo, double hi, bool density);

// Compute a 2-D histogram from two equal-length 1-D tensors `a` (x) and
// `b` (y) over independent uniform grids.
//
// Returns {counts, edges} where:
//   counts — 2-D tensor of shape (bins_a, bins_b), dtype F64.  Element [i,j]
//            counts pairs (a_k, b_k) that fall in the i-th x-bin and j-th
//            y-bin.
//   edges  — 1-D tensor of shape (bins_a+1 + bins_b+1,) concatenating the
//            x-edges followed by the y-edges.
//
// If `density` is true, each cell is normalised by (step_a * step_b * N).
LUCID_API std::vector<TensorImplPtr> histogram2d_op(const TensorImplPtr& a,
                                                    const TensorImplPtr& b,
                                                    std::int64_t bins_a,
                                                    std::int64_t bins_b,
                                                    double lo_a,
                                                    double hi_a,
                                                    double lo_b,
                                                    double hi_b,
                                                    bool density);

// Compute an N-dimensional histogram from the rows of a 2-D tensor `a` of
// shape (N, D).  Each column represents one dimension of the data.
//
// `bins`   — number of bins for each of the D dimensions.
// `ranges` — (lo, hi) range for each dimension.
//
// Returns {counts, edges} where:
//   counts — D-dimensional tensor whose shape equals `bins`, dtype F64.
//   edges  — 1-D tensor of shape (sum(bins[d]+1),) concatenating the edge
//            arrays for all D dimensions in order.
//
// If `density` is true, each cell is normalised by (cell_volume * N).
LUCID_API std::vector<TensorImplPtr> histogramdd_op(const TensorImplPtr& a,
                                                    std::vector<std::int64_t> bins,
                                                    std::vector<std::pair<double, double>> ranges,
                                                    bool density);

}  // namespace lucid
