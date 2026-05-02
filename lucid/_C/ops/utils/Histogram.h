#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API std::vector<TensorImplPtr>
histogram_op(const TensorImplPtr& a, std::int64_t bins, double lo, double hi, bool density);

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
