#pragma once

// =====================================================================
// Histogram ops — bin counting on the flattened input or along a multi-D
// joint distribution. Forward only (no autograd).
//
//   histogram(a, bins, lo, hi, density)            — 1-D
//   histogram2d(a, b, bins_a, bins_b, lo, hi, ...) — 2-D joint
//   histogramdd(a, bins, lo, hi, density)          — N-D joint over last axis
//
// All return (counts, edges). For 1-D `histogram`, edges is a single
// 1-D tensor of length bins+1. For 2-D and N-D, edges is a list packed as
// a 1-D tensor — see the Python wrapper for unpacking semantics.
// =====================================================================

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API std::vector<TensorImplPtr> histogram_op(
    const TensorImplPtr& a, std::int64_t bins, double lo, double hi, bool density);

LUCID_API std::vector<TensorImplPtr> histogram2d_op(const TensorImplPtr& a,
                                                    const TensorImplPtr& b,
                                                    std::int64_t bins_a,
                                                    std::int64_t bins_b,
                                                    double lo_a,
                                                    double hi_a,
                                                    double lo_b,
                                                    double hi_b,
                                                    bool density);

LUCID_API std::vector<TensorImplPtr> histogramdd_op(const TensorImplPtr& a,
                                                    std::vector<std::int64_t> bins,
                                                    std::vector<std::pair<double, double>> ranges,
                                                    bool density);

}  // namespace lucid
