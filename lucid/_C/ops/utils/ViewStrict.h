// lucid/_C/ops/utils/ViewStrict.h
//
// ``view``: the reshape that never copies.  Implemented in View.cpp beside
// reshape_op, whose view rule it shares.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Reshape that never copies: the result always shares ``a``'s storage.
//
// The same view :func:`reshape_op` makes when ``a``'s elements read in place
// as ``new_shape`` — a dense tensor, on either device, or a strided CPU view
// whose new shape only drops or inserts size-1 axes.  Where reshape_op would
// copy instead (``x.t().view(-1)``), this raises.
//
// Raises
// ------
// ShapeMismatch
//     The element counts differ, or ``-1`` cannot be inferred.
// LucidError
//     ``a`` cannot be viewed as ``new_shape`` without a copy.
LUCID_API TensorImplPtr view_op(const TensorImplPtr& a, const std::vector<std::int64_t>& new_shape);

}  // namespace lucid
