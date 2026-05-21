// lucid/_C/ops/ufunc/Var.h
//
// Public entry point for the variance reduction op.  The backward node
// (``VarBackward``) lives entirely inside ``Var.cpp`` in an anonymous
// namespace and is intentionally private — callers should depend only
// on :func:`var_op`.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the biased variance of ``a`` along ``axes``.
//
// Uses the population estimator
// $\sigma^2 = \frac{1}{N} \sum_i (x_i - \bar{x})^2$ — i.e. the
// denominator is $N$, not $N - 1$.  This corresponds to the reference
// framework's ``unbiased=False`` (population) variant; the unbiased /
// sample variant is not exposed by this entry point.  The standard
// deviation :func:`std_op` is built directly on top by composing
// ``sqrt`` with this op so the autograd chain is automatic.
//
// During the forward pass the mean is computed once with
// ``keepdims=true`` and broadcast to the full input shape so that
// ``VarBackward::apply`` can compute $x - \bar{x}$ without a second
// broadcast call.  Both the input ``x`` and the broadcast mean are
// saved on the backward node.
//
// Math
// ----
// $$
//   \sigma^2 = \frac{1}{N} \sum_{i \in \text{axes}}
//   (x_i - \bar{x})^2, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \frac{2}{N}(x_i - \bar{x})\cdot
//   \mathrm{broadcast}\!\left(\frac{\partial \mathcal{L}}
//   {\partial \sigma^2}\right).
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of float dtype.  Non-null.
// axes : std::vector<int>
//     Axes to reduce.  May be empty (reduce all axes).  Negative
//     indices wrap around ``ndim``.
// keepdims : bool
//     If ``true``, retains reduced dims as size-1 entries; otherwise
//     they are collapsed.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with the reduced shape.  Autograd is wired when
//     ``a->requires_grad()`` is true and ``GradMode`` is enabled.
//
// Shape
// -----
// If ``keepdims`` is ``false``, the reduced axes are removed; if
// ``true``, they become size-$1$.
//
// Notes
// -----
// Internal dispatch: ``IBackend::variance`` — Accelerate on CPU, MLX
// on GPU.  The CPU implementation uses Welford-style accumulation for
// numerical stability on large $N$.  The empty-axis edge case
// (``reduced == 0``) clamps the count to $1$ to avoid a division by
// zero in the backward node; in that pathological case the gradient is
// not statistically meaningful but is well-defined.
//
// See Also
// --------
// :func:`std_op` — square root of the variance.
LUCID_API TensorImplPtr var_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

}  // namespace lucid
