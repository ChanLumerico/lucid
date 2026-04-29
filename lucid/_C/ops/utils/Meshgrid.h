#pragma once

// =====================================================================
// meshgrid: Cartesian product on broadcast axes (numpy/PyTorch convention).
//   indexing="ij" returns each output with shape ⊗(L_i) in input order;
//   "xy" swaps the first two axes (numpy default).
// =====================================================================

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// Meshgrid.
LUCID_API std::vector<TensorImplPtr> meshgrid_op(const std::vector<TensorImplPtr>& xs,
                                                 bool indexing_xy);

}  // namespace lucid
