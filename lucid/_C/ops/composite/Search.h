// lucid/_C/ops/composite/Search.h
//
// Sorted-array binary search ops:
//   searchsorted(sorted_1d, values)
//       — for each query, return the leftmost (right=false) or rightmost
//         (right=true) insertion point in ``sorted_1d`` (or per row of an
//         N-D ``sorted_1d`` whose leading N-1 dimensions match ``values``).
//   bucketize(values, boundaries)
//       — alias of searchsorted with the argument order flipped.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Per-query insertion point into a sorted 1-D (or row-wise N-D) reference.
//
// Evaluates each query via binary search in $O(\log N)$ time and $O(1)$
// auxiliary memory per query, ordering ``NaN`` after ``+inf`` to match the
// reference.  Not differentiable — integer output.
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
//     1-D ascending-sorted reference of length $N$, or an N-D tensor whose
//     leading $N - 1$ dimensions match ``values``.
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
//     If either input is null or the shapes are incompatible.
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
