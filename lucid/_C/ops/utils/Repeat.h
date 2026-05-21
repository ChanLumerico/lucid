// lucid/_C/ops/utils/Repeat.h
//
// Declares the tiling operations repeat and tile.  Both replicate input data
// along one or more dimensions and carry autograd support for floating-point
// inputs.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Repeat the contents of a tensor along a single axis.
//
// Produces an output whose ``axis`` dimension is ``repeats`` times larger
// than the input's; every other dimension is unchanged.  Each element along
// ``axis`` is replicated as a contiguous block of ``repeats`` copies in the
// output (this is interleaved repetition, distinct from :func:`tile_op`).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// repeats : int64_t
//     Number of times to repeat along ``axis``.  Must be > 0.
// axis : int
//     Axis along which to repeat.  Supports negative indexing.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``a.shape`` with the ``axis`` dimension scaled by
//     ``repeats``.
//
// Shape
// -----
// ``out.shape[axis] = a.shape[axis] * repeats``; all other dims unchanged.
//
// Math
// ----
// $$ y_{[\ldots, i_a, \ldots]} = x_{[\ldots, \lfloor i_a / r \rfloor, \ldots]} $$
// where $r = $ ``repeats`` and $i_a$ is the index along ``axis``.
//
// Notes
// -----
// Backward sums the repeated gradient blocks back into the original shape via
// ``Dispatcher::repeat_backward``.  Schema name ``"repeat"``,
// ``AmpPolicy::KeepInput``.
//
// See Also
// --------
// :func:`tile_op` — multi-axis tiling; different element ordering.
LUCID_API TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis);

// Tile a tensor by repeating it along multiple axes.
//
// ``reps[d]`` specifies how many times to repeat along dimension ``d``.  If
// ``reps`` is longer than ``a.rank()``, the input is conceptually left-padded
// with size-1 dimensions before tiling, so the output rank equals
// ``reps.size()``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// reps : vector<int64_t>
//     Per-dimension repetition counts.  Each entry must be >= 1.  If
//     ``reps.size() > a.rank()``, leading size-1 axes are virtually prepended
//     to the input.
//
// Returns
// -------
// TensorImplPtr
//     Tiled tensor with rank ``max(reps.size(), a.rank())`` and
//     ``out.shape[d] = a_padded.shape[d] * reps[d]``.
//
// Shape
// -----
// Let ``r = reps.size()`` and ``n = a.rank()``.  After left-padding ``a`` with
// ``max(r - n, 0)`` size-1 dims:
// ``out.shape[d] = a_padded.shape[d] * reps[d]`` for ``d in [0, max(r, n))``.
//
// Math
// ----
// $$ y_{[i_0, \ldots, i_{r-1}]} = x_{[i_0 \bmod D_0, \ldots, i_{r-1} \bmod D_{r-1}]} $$
// where $D_d$ is the input's effective size along dimension $d$.
//
// Notes
// -----
// Backward sums the tiled gradient blocks via ``Dispatcher::tile_backward``,
// which uses the padded input shape and the repetition counts to locate each
// contribution.  Schema name ``"tile"``, ``AmpPolicy::KeepInput``.
//
// Examples
// --------
// ``tile_op([1, 2, 3], {2})`` produces ``[1, 2, 3, 1, 2, 3]``;
// ``tile_op([[1, 2]], {2, 3})`` produces a 2x6 tensor
// ``[[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]``.
//
// See Also
// --------
// :func:`repeat_op` — single-axis, interleaved variant.
LUCID_API TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps);

}  // namespace lucid
