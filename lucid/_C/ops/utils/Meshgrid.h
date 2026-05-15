// lucid/_C/ops/utils/Meshgrid.h
//
// Declares the meshgrid op, which generates an N-dimensional coordinate grid
// from a list of 1-D input tensors.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Produce N output grids from N 1-D input tensors, where the i-th output
// broadcasts the i-th input across all dimensions except its "carry" axis.
//
// `indexing_xy` controls the axis assignment for the first two tensors:
//   false ("ij" indexing): output[i] varies along axis i.
//   true  ("xy" indexing): output[0] varies along axis 1 (x), and
//                          output[1] varies along axis 0 (y); all others
//                          follow the ij convention.
//
// All inputs must be 1-D and share the same dtype and device.  The output
// shape is determined by the sizes of all inputs (possibly swapped for the
// first two when indexing_xy is true).
//
// Backward: for each output gradient, sum over all axes except the carry
// axis to recover the gradient w.r.t. the corresponding 1-D input.
LUCID_API std::vector<TensorImplPtr> meshgrid_op(const std::vector<TensorImplPtr>& xs,
                                                 bool indexing_xy);

}  // namespace lucid
