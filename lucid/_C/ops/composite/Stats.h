// lucid/_C/ops/composite/Stats.h
//
// Statistical / combinatorial helpers built atop primitive ops.
//
//   histc(a, bins, lo, hi)     — counts-only wrapper around ``histogram``
//                                with auto-range when lo == hi
//   cartesian_prod(tensors...) — meshgrid + flatten + stack to enumerate
//                                every combination across 1-D inputs

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Histogram counts over $[\mathrm{lo}, \mathrm{hi})$ with ``bins`` uniform buckets.
//
// Composite — counts-only wrapper around :func:`histogram_op`.  When
// ``lo == hi`` the range is auto-derived from the input via :func:`min_op`
// and :func:`max_op`, with the degenerate ``min == max`` case bumped by
// $+1$ so the bins remain well-defined (matches reference framework
// behaviour).  Not differentiable — integer-valued output.
//
// Math
// ----
// Let $w = (\mathrm{hi} - \mathrm{lo}) / \mathrm{bins}$.  Then for each
// bin index $k \in [0, \mathrm{bins})$,
// $$
//   y_k = \#\!\left\{\, i : \mathrm{lo} + k w \le a_i < \mathrm{lo} + (k+1) w \,\right\}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of arbitrary shape.
// bins : int64
//     Number of histogram bins.  Must be positive.
// lo : double
//     Inclusive lower edge.  If ``lo == hi``, auto-range from
//     ``min(a)`` to ``max(a)``.
// hi : double
//     Exclusive upper edge.
//
// Returns
// -------
// TensorImplPtr
//     1-D tensor of length ``bins`` holding integer counts.
//
// Raises
// ------
// Failure
//     If ``a`` is null.
//
// Notes
// -----
// Values outside ``[lo, hi)`` are dropped silently by the underlying
// :func:`histogram_op` — no overflow / underflow bins are returned.
//
// See Also
// --------
// :func:`histogram_op` — full version that also returns bin edges.
LUCID_API TensorImplPtr histc_op(const TensorImplPtr& a, std::int64_t bins, double lo, double hi);

// Enumerate every combination across a list of 1-D tensors.
//
// Composite over :func:`meshgrid_op` (with ``ij`` indexing) +
// :func:`reshape_op` + :func:`stack_op`.  Builds the n-D grid, flattens
// each component, and stacks the flattened columns along a new trailing
// axis to produce a ``(N, D)`` matrix whose rows enumerate the full
// Cartesian product in row-major (ij) order.  Gradient flows through
// ``MeshgridBackward``, ``ReshapeBackward``, and ``StackBackward``.
//
// Math
// ----
// For input tensors $t^{(0)}, \ldots, t^{(D-1)}$ of lengths
// $n_0, \ldots, n_{D-1}$, the output rows are
// $$
//   \left( t^{(0)}_{i_0}, t^{(1)}_{i_1}, \ldots, t^{(D-1)}_{i_{D-1}} \right),
//   \qquad (i_0, \ldots, i_{D-1}) \in \prod_d [0, n_d)
// $$
// in row-major (lex) order with $i_0$ varying slowest.
//
// Parameters
// ----------
// tensors : vector<TensorImplPtr>
//     Non-empty list of 1-D tensors.  Lengths may differ.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``(N, D)`` where ``N = prod(lengths)`` and ``D``
//     is the number of inputs.  Dtype matches the inputs (must agree).
//
// Raises
// ------
// Failure
//     If the input list is empty, any element is null, or any element is
//     not 1-D.
//
// Examples
// --------
// ``cartesian_prod_op({[1, 2], [10, 20, 30]})`` returns a ``(6, 2)``
// tensor: ``[[1, 10], [1, 20], [1, 30], [2, 10], [2, 20], [2, 30]]``.
//
// See Also
// --------
// :func:`meshgrid_op`, :func:`stack_op`.
LUCID_API TensorImplPtr cartesian_prod_op(const std::vector<TensorImplPtr>& tensors);

}  // namespace lucid
