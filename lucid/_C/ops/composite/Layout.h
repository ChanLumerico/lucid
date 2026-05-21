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

// Move dimensions from ``source`` to ``destination`` positions.
//
// Composite over :func:`permute_op`.  Builds an explicit permutation in
// which every axis listed in ``source`` is placed at the matching index in
// ``destination``; axes not listed preserve their original relative order
// and fill the remaining slots left to right.  No new backward node —
// gradient flows through ``PermuteBackward``.
//
// Math
// ----
// If $\pi$ is the constructed permutation of $[0, n)$, then
// $$
//   y_{i_0, \ldots, i_{n-1}} = a_{i_{\pi(0)}, \ldots, i_{\pi(n-1)}}
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor.
// source : vector<int>
//     Axes of ``a`` to move.  Negative values wrap modulo ``a.ndim``.
//     Must be unique.
// destination : vector<int>
//     Target positions for the moved axes, parallel to ``source``.
//     Negative values wrap.  Must be unique and have the same length as
//     ``source``.
//
// Returns
// -------
// TensorImplPtr
//     A view (or contiguous copy, per the primitive's policy) of ``a``
//     with the requested axis order.
//
// Raises
// ------
// Failure
//     If ``a`` is null, the lists differ in length, or either list
//     contains duplicates.
// IndexError
//     If any axis is out of range.
//
// Examples
// --------
// ``movedim_op(a, {0}, {-1})`` moves the leading axis to the trailing
// position — equivalent to a cyclic left rotation of axes.
//
// See Also
// --------
// :func:`permute_op` — underlying primitive.
LUCID_API TensorImplPtr movedim_op(const TensorImplPtr& a,
                                   const std::vector<int>& source,
                                   const std::vector<int>& destination);

// Inverse of ``flatten`` — split ``dim`` into ``sizes``.
//
// Composite over :func:`reshape_op`.  The product of ``sizes`` must equal
// ``a.shape[dim]``; the surrounding axes are preserved verbatim.  Pure
// view operation when the input is contiguous; gradient flows through
// ``ReshapeBackward``.
//
// Math
// ----
// Let ``a`` have shape $(s_0, \ldots, s_{d-1}, s_d, s_{d+1}, \ldots)$ and
// ``sizes`` $= (m_0, \ldots, m_{k-1})$ with $\prod_i m_i = s_d$.  Then the
// output has shape
// $$
//   (s_0, \ldots, s_{d-1}, m_0, \ldots, m_{k-1}, s_{d+1}, \ldots)
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor of rank $\ge 1$.
// dim : int
//     Axis to expand.  Negative values wrap modulo ``a.ndim``.
// sizes : Shape
//     Replacement extents.  Their product must equal ``a.shape[dim]``.
//
// Returns
// -------
// TensorImplPtr
//     Reshaped tensor whose total ``numel`` matches ``a``.
//
// Raises
// ------
// Failure
//     If ``a`` is null or $\prod \mathrm{sizes} \ne \mathrm{a.shape}[\mathrm{dim}]$.
// IndexError
//     If ``dim`` is out of range.
//
// See Also
// --------
// :func:`reshape_op` — the underlying primitive operates on the full shape.
LUCID_API TensorImplPtr unflatten_op(const TensorImplPtr& a, int dim, const Shape& sizes);

}  // namespace lucid
