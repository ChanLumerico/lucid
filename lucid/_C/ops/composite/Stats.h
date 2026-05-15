// lucid/_C/ops/composite/Stats.h
//
// Statistical / combinatorial helpers built atop primitive ops.
//
//   histc(a, bins, lo, hi)     — counts-only wrapper around ``histogram``
//                                with auto-range when lo == hi
//   cartesian_prod(tensors...) — meshgrid + flatten + stack to enumerate
//                                every combination across 1-D inputs

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr histc_op(const TensorImplPtr& a, std::int64_t bins, double lo, double hi);

LUCID_API TensorImplPtr cartesian_prod_op(const std::vector<TensorImplPtr>& tensors);

}  // namespace lucid
