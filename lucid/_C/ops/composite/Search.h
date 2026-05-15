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

LUCID_API TensorImplPtr searchsorted_op(const TensorImplPtr& sorted_1d,
                                        const TensorImplPtr& values,
                                        bool right);

LUCID_API TensorImplPtr bucketize_op(const TensorImplPtr& values,
                                     const TensorImplPtr& boundaries,
                                     bool right);

}  // namespace lucid
