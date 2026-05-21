// lucid/_C/ops/ufunc/Trace.h
//
// Public entry point for the matrix trace operation.  The backward
// node (``TraceBackward``) is defined entirely inside ``Trace.cpp`` in
// an anonymous namespace and is intentionally private — callers should
// depend only on :func:`trace_op`.
//
// Trace is the sum of main-diagonal elements of a matrix.  Lucid's
// implementation supports both strict 2-D matrices and batched
// inputs of shape ``[..., m, n]``; autograd is currently wired only
// for the 2-D case because the backend's ``trace_backward`` scatter
// kernel does not yet handle the batched layout.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the trace (sum of main-diagonal elements) of a matrix or a
// batch of matrices.
//
// For a 2-D input ``a`` of shape ``[m, n]`` the result is a scalar
// $\mathrm{tr}(a) = \sum_i a_{ii}$ where $i$ ranges over
// $\min(m, n)$.  For a higher-rank input of shape
// ``[b_0, ..., b_{k-1}, m, n]`` the last two dimensions are
// contracted: each $[m, n]$ slice produces one scalar, and the result
// has shape ``[b_0, ..., b_{k-1}]``.
//
// Backward (2-D only) scatters $\partial \mathcal{L}/\partial y$ to
// each diagonal position of a zero-filled matrix of the same shape
// as ``a``:
// $\partial \mathcal{L}/\partial a_{ij} =
// \mathbb{1}[i = j]\cdot \partial \mathcal{L}/\partial y$.
// For batched inputs (``ndim > 2``) the output is returned without an
// autograd edge — attempts to backpropagate through batched
// :func:`trace_op` will silently produce no gradient.
//
// Math
// ----
// 2-D forward and backward:
// $$
//   y = \sum_{i} a_{ii}, \qquad
//   \frac{\partial \mathcal{L}}{\partial a_{ij}} =
//   \mathbb{1}[i = j]\cdot
//   \frac{\partial \mathcal{L}}{\partial y}.
// $$
//
// Batched forward (no offset, no autograd):
// $$
//   y_{b_0 \dots b_{k-1}} = \sum_{i}
//   a_{b_0 \dots b_{k-1}, i, i}.
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any dtype, with ``ndim >= 2``.  Non-null.
//
// Returns
// -------
// TensorImplPtr
//     Scalar (``[]``) for 2-D inputs, or a tensor of shape
//     ``a.shape[2:]`` for batched inputs.
//
// Raises
// ------
// Error
//     If ``a`` is null or has ``ndim < 2``.
//
// Shape
// -----
// - Input  ``[m, n]``                          → output ``[]``
// - Input  ``[b_0, ..., b_{k-1}, m, n]``       → output
//   ``[b_0, ..., b_{k-1}]``
//
// Notes
// -----
// Diagonal offset (analogous to NumPy's ``trace(offset=...)``) is not
// currently exposed; only the main diagonal is summed.  Autograd is
// wired only when ``ndim == 2`` — the batched path returns a plain
// non-differentiable tensor by design.
LUCID_API TensorImplPtr trace_op(const TensorImplPtr& a);

}  // namespace lucid
