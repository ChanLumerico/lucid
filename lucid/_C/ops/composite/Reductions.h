// lucid/_C/ops/composite/Reductions.h
//
// Reduction operations expressed as compositions of primitive reductions.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Numerically-stable $\log \sum_i \exp(x_i)$ via the max-shift trick.
//
// Composite over :func:`max_op` + :func:`sub_op` + :func:`exp_op` +
// :func:`sum_op` + :func:`log_op` + :func:`add_op` (+ :func:`squeeze_op`).
// Subtracting the per-reduction-group max keeps the largest exponent
// argument at $0$ so the rest underflow rather than overflow; the offset
// is added back outside the log.  Gradient flows through ``MaxBackward``,
// ``SubBackward``, ``ExpBackward``, ``SumBackward``, ``LogBackward``, and
// ``AddBackward`` automatically — no dedicated backward node is registered.
//
// Math
// ----
// $$
//   y = m + \ln \!\left( \sum_{i \in \mathrm{axes}} \exp(x_i - m) \right),
//   \qquad m = \max_{i \in \mathrm{axes}} x_i
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axes : vector<int>
//     Axes to reduce.  Negative values wrap modulo ``a.ndim``.  An empty
//     list reduces over all axes per the primitive's convention.
// keepdims : bool
//     If ``true``, retain size-1 placeholders at the reduced positions.
//     Otherwise, the reduced axes are squeezed off (in descending order
//     so each remaining index stays valid through the sequential squeeze
//     loop).
//
// Returns
// -------
// TensorImplPtr
//     Reduced tensor.  Dtype matches ``a``.
//
// Notes
// -----
// The intermediate :func:`max_op` is computed with ``keepdims=true`` so
// the broadcast subtract is well defined; the optional axis collapse is
// done at the end on the keep-dim result.
//
// See Also
// --------
// :func:`logaddexp_op` — two-operand elementwise variant.
LUCID_API TensorImplPtr logsumexp_op(const TensorImplPtr& a,
                                     const std::vector<int>& axes,
                                     bool keepdims);

}  // namespace lucid
