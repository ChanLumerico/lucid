// lucid/_C/ops/composite/Search.h
//
// Sorted-array search ops built on top of broadcasting comparisons.  The
// trick: counting how many elements of a sorted reference are strictly less
// than (or less-than-or-equal to) each query value is exactly the index
// where the query would be inserted to keep the array sorted.  No specialised
// binary-search kernel needed — broadcast + compare + reduce primitives
// already produce the answer.
//
//   searchsorted(sorted_1d, values)
//       — for each query, return the leftmost (right=false) or rightmost
//         (right=true) insertion point in ``sorted_1d``.
//   bucketize(values, boundaries)
//       — alias of searchsorted with the argument order flipped.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Per-query insertion point into a sorted 1-D reference.
//
// Composite over :func:`reshape_op` + :func:`broadcast_to_op` +
// (:func:`less_op` | :func:`less_equal_op`) + :func:`astype_op` +
// :func:`sum_op`.  The 1-D reference is broadcast to
// ``[1, ..., 1, N]`` and the queries to ``S + [1]``; both expand to the
// full ``S + [N]`` grid, the comparison is reduced over the trailing axis,
// and the bool $\to$ F32 $\to$ I64 cast chain produces the standard
// ``searchsorted`` dtype.  Not differentiable — integer output.
//
// Math
// ----
// Let $s$ be the sorted reference and $v$ the query tensor.  Then
// $$
//   y_{\ldots, j} = \sum_{i=0}^{N-1}
//   \mathbb{1}\!\left[ s_i \prec v_{\ldots, j} \right]
// $$
// with $\prec$ being $<$ when ``right = false`` (leftmost insertion) and
// $\le$ when ``right = true`` (rightmost insertion).
//
// Parameters
// ----------
// sorted_1d : TensorImplPtr
//     1-D ascending-sorted reference of length $N$.
// values : TensorImplPtr
//     Query tensor of arbitrary shape ``S``.
// right : bool
//     If ``true``, returns the rightmost (post-equal) insertion point;
//     otherwise the leftmost.
//
// Returns
// -------
// TensorImplPtr
//     ``int64`` tensor of shape ``S`` whose entries lie in $[0, N]$.
//
// Raises
// ------
// Failure
//     If either input is null or ``sorted_1d`` is not 1-D.
//
// Notes
// -----
// The reduction kernel only supports float dtypes; the bool comparison is
// cast to F32 for the sum, then back to I64 to match the canonical
// ``searchsorted`` dtype.
//
// See Also
// --------
// :func:`bucketize_op` — same operation with the argument order flipped.
LUCID_API TensorImplPtr searchsorted_op(const TensorImplPtr& sorted_1d,
                                        const TensorImplPtr& values,
                                        bool right);

// Bucket-index lookup: alias of :func:`searchsorted_op` with the operand
// order flipped to match the bucket-first convention.
//
// Composite — delegates directly to :func:`searchsorted_op(boundaries,
// values, right)`.  Not differentiable.
//
// Math
// ----
// $$
//   y_{\ldots, j} = \sum_{i} \mathbb{1}\!\left[ b_i \prec v_{\ldots, j} \right]
// $$
// with $b$ = ``boundaries`` and the comparison $\prec$ selected by
// ``right`` per :func:`searchsorted_op`.
//
// Parameters
// ----------
// values : TensorImplPtr
//     Query tensor of arbitrary shape.
// boundaries : TensorImplPtr
//     1-D ascending-sorted boundary tensor.
// right : bool
//     If ``true``, queries equal to a boundary land in the higher bucket.
//
// Returns
// -------
// TensorImplPtr
//     ``int64`` tensor of the same shape as ``values`` whose entries are
//     bucket indices in $[0, \mathrm{len}(\mathrm{boundaries})]$.
//
// See Also
// --------
// :func:`searchsorted_op`.
LUCID_API TensorImplPtr bucketize_op(const TensorImplPtr& values,
                                     const TensorImplPtr& boundaries,
                                     bool right);

}  // namespace lucid
