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

// Build N broadcast coordinate grids from N 1-D input tensors.
//
// Given inputs $x_0, x_1, \ldots, x_{N-1}$ with lengths
// $L_0, L_1, \ldots, L_{N-1}$, produces N output tensors all of the same
// rank-N shape; the $i$-th output broadcasts $x_i$ across every axis
// except its "carry" axis, where it varies.  This is the
// multi-dimensional analogue of NumPy's ``np.meshgrid``.
//
// Parameters
// ----------
// xs : vector<TensorImplPtr>
//     One or more 1-D input tensors.  All must share the same dtype and
//     device.
// indexing_xy : bool
//     Axis convention for the first two inputs:
//
//     * ``false`` — ``"ij"`` (matrix) indexing.  Output ``i`` varies along
//       axis ``i``; output shape is ``(L_0, L_1, ..., L_{N-1})``.
//     * ``true`` — ``"xy"`` (Cartesian) indexing.  Output 0 varies along
//       axis 1 (x) and output 1 varies along axis 0 (y); outputs
//       ``i >= 2`` follow the ``ij`` convention.  Output shape is
//       ``(L_1, L_0, L_2, ..., L_{N-1})``.
//
// Returns
// -------
// vector<TensorImplPtr>
//     N coordinate grids, one per input, each of rank N.
//
// Math
// ----
// For ``ij`` indexing:
// $$\mathrm{out}_i[k_0, k_1, \ldots, k_{N-1}] = x_i[k_i]$$
//
// Notes
// -----
// Backward sums each output gradient over every axis except the carry
// axis to recover the gradient w.r.t. the corresponding 1-D input.  All
// inputs must be 1-D; mismatched ranks raise.
//
// Examples
// --------
// >>> X, Y = meshgrid([x, y], indexing_xy=true)  // 'xy' convention
// >>> X.shape == (len(y), len(x))
// True
//
// See Also
// --------
// Concat : Stack the resulting grids along a new leading axis.
LUCID_API std::vector<TensorImplPtr> meshgrid_op(const std::vector<TensorImplPtr>& xs,
                                                 bool indexing_xy);

}  // namespace lucid
