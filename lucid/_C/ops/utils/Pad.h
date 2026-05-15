// lucid/_C/ops/utils/Pad.h
//
// Declares the constant-fill padding operation.  The op supports padding each
// dimension independently with (before, after) widths and fills the added
// elements with a caller-supplied constant value.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Pad `a` with `constant` fill on each side of each dimension.
//
// `pad_width` is a vector of (before, after) pairs with length equal to a's
// rank.  The output shape is: out_shape[d] = a.shape[d] + pad_width[d].first
// + pad_width[d].second for each dimension d.
//
// Backward: for each dimension in order, slice out the original (unpadded)
// region by discarding pad_width[d].first leading and pad_width[d].second
// trailing elements.  The slices are applied sequentially, dimension by
// dimension, using the pad_width offsets recorded in PadBackward.
LUCID_API TensorImplPtr pad_op(const TensorImplPtr& a,
                               std::vector<std::pair<std::int64_t, std::int64_t>> pad_width,
                               double constant);

}  // namespace lucid
