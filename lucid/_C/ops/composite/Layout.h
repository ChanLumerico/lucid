// lucid/_C/ops/composite/Layout.h
//
// Shape rearrangement helpers expressed as compositions of ``permute`` and
// ``reshape``.  Both ops are differentiable, so the gradient flows back
// through the underlying primitive without a new backward node.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Shape.h"
#include "../../core/fwd.h"

namespace lucid {

// Move dimensions from ``source`` to ``destination`` positions.  Both args
// hold the same number of indices; the remaining axes preserve their
// original relative order and fill the gaps left to right.  Composes with
// ``permute_op``.
LUCID_API TensorImplPtr movedim_op(const TensorImplPtr& a,
                                   const std::vector<int>& source,
                                   const std::vector<int>& destination);

// Inverse of ``flatten`` — split ``dim`` into ``sizes``.  The product of
// ``sizes`` must equal the original size at ``dim``.  Composes with
// ``reshape_op``.
LUCID_API TensorImplPtr unflatten_op(const TensorImplPtr& a, int dim, const Shape& sizes);

}  // namespace lucid
