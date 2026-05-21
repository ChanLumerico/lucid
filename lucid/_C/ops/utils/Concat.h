// lucid/_C/ops/utils/Concat.h
//
// Public API for concatenation, stacking, splitting, chunking, and unbinding
// operations.  All functions allocate fresh output tensors and wire autograd
// nodes where required.  The corresponding backward passes are defined in
// Concat.cpp and are not exposed here because they are purely internal.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Concatenate a sequence of tensors along an existing axis.
//
// All inputs must share dtype, device, and rank; every dimension except
// ``axis`` must agree exactly.  The output reuses freshly allocated storage —
// no input buffer is aliased — and the autograd node saves only per-input
// shapes so backward can split the gradient back into one slice per input.
//
// Parameters
// ----------
// xs : vector<TensorImplPtr>
//     Non-empty list of input tensors with matching dtype, device, and rank.
//     All dimensions except ``axis`` must be identical across inputs.
// axis : int
//     Axis along which to concatenate.  Supports negative indexing
//     (wrapped to ``axis + ndim``).  Must satisfy ``-ndim <= axis < ndim``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of shape ``(..., \sum_i N_i, ...)`` where the ``axis``
//     dimension is the sum of each input's ``axis`` dimension.
//
// Shape
// -----
// Inputs of shape ``(D_0, ..., D_{axis-1}, N_i, D_{axis+1}, ..., D_{ndim-1})``
// produce output of shape
// ``(D_0, ..., D_{axis-1}, \sum_i N_i, D_{axis+1}, ..., D_{ndim-1})``.
//
// Math
// ----
// $$ y_{[\ldots,\, k,\, \ldots]} = x^{(i)}_{[\ldots,\, k - \sum_{j<i} N_j,\, \ldots]} $$
// where $i$ is the unique input index satisfying
// $\sum_{j<i} N_j \le k < \sum_{j \le i} N_j$.
//
// Raises
// ------
// DtypeMismatch
//     If inputs disagree on dtype.
// DeviceMismatch
//     If inputs reside on different devices.
// ShapeMismatch
//     If inputs disagree on any non-concat dimension, or differ in rank.
//
// Notes
// -----
// Backward slices the incoming gradient back into per-input pieces using the
// saved size of each input along ``axis``; the schema is
// ``"concatenate"`` with ``AmpPolicy::KeepInput``.
//
// See Also
// --------
// :func:`stack_op`, :func:`split_op`, :func:`hstack_op`, :func:`vstack_op`.
LUCID_API TensorImplPtr concatenate_op(const std::vector<TensorImplPtr>& xs, int axis);

// Stack a list of same-shape tensors along a new axis.
//
// Inserts a fresh size-``len(xs)`` dimension at position ``axis`` of the
// output, increasing the rank by one.  All inputs must have identical shape,
// dtype, and device.
//
// Parameters
// ----------
// xs : vector<TensorImplPtr>
//     Non-empty list of tensors with identical shape, dtype, and device.
// axis : int
//     Insertion position for the new dimension.  Supports negative indexing
//     against the output rank ``ndim + 1`` (so ``axis = -1`` appends).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor with rank ``ndim + 1`` and a new dimension of size
//     ``len(xs)`` inserted at ``axis``.
//
// Shape
// -----
// Inputs of shape ``(D_0, ..., D_{ndim-1})`` stacked along ``axis = k`` yield
// output shape ``(D_0, ..., D_{k-1}, len(xs), D_k, ..., D_{ndim-1})``.
//
// Raises
// ------
// ShapeMismatch, DtypeMismatch, DeviceMismatch
//     On any disagreement between inputs.
//
// Notes
// -----
// Backward slices each size-1 hyperplane along the new axis and squeezes it
// to recover the gradient for the corresponding input.  Schema name
// ``"stack"``, ``AmpPolicy::KeepInput``.
//
// See Also
// --------
// :func:`concatenate_op`, :func:`unbind_op`.
LUCID_API TensorImplPtr stack_op(const std::vector<TensorImplPtr>& xs, int axis);

// Horizontally stack a sequence of tensors.
//
// Equivalent to NumPy ``hstack``: for 1-D inputs the result concatenates
// along axis 0; for tensors with rank >= 2 the concatenation occurs along
// axis 1.  Tensors must agree on every non-horizontal dimension.
//
// Parameters
// ----------
// xs : vector<TensorImplPtr>
//     Non-empty list of tensors with the same rank, dtype, and device.
//
// Returns
// -------
// TensorImplPtr
//     Horizontally concatenated tensor.
//
// Examples
// --------
// 1-D inputs ``[a, b, c]`` and ``[d, e]`` produce ``[a, b, c, d, e]``.
// 2-D inputs of shapes ``(R, C_i)`` produce shape ``(R, \sum_i C_i)``.
//
// See Also
// --------
// :func:`vstack_op`, :func:`concatenate_op`.
LUCID_API TensorImplPtr hstack_op(const std::vector<TensorImplPtr>& xs);

// Vertically stack a sequence of tensors.
//
// Equivalent to NumPy ``vstack``: 1-D inputs of length ``N`` are promoted to
// row vectors and stacked into a 2-D tensor of shape ``(len(xs), N)``;
// inputs with rank >= 2 are concatenated along axis 0.
//
// Parameters
// ----------
// xs : vector<TensorImplPtr>
//     Non-empty list of tensors.  For rank >= 2, every dim except axis 0
//     must agree; for 1-D inputs, lengths must agree.
//
// Returns
// -------
// TensorImplPtr
//     Vertically stacked tensor with rank ``max(2, input_rank)``.
//
// See Also
// --------
// :func:`hstack_op`, :func:`stack_op`.
LUCID_API TensorImplPtr vstack_op(const std::vector<TensorImplPtr>& xs);

// Split a tensor into a fixed number of equal pieces along an axis.
//
// The size of ``a`` along ``axis`` must be divisible by ``num_splits``.
// Each output piece is wired with a ``SplitSliceBackward`` node that
// scatters its gradient back into the correct slice of the input gradient
// via ``insert_axis_slice``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// num_splits : int64_t
//     Number of equal pieces.  Must divide ``a.shape[axis]`` exactly.
// axis : int
//     Axis to split along.  Supports negative indexing.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``num_splits`` tensors, each with shape identical to ``a`` except the
//     ``axis`` dimension equals ``a.shape[axis] / num_splits``.
//
// Raises
// ------
// ShapeMismatch
//     If ``a.shape[axis]`` is not divisible by ``num_splits``.
//
// See Also
// --------
// :func:`split_at_op`, :func:`chunk_op`, :func:`unbind_op`.
LUCID_API std::vector<TensorImplPtr>
split_op(const TensorImplPtr& a, std::int64_t num_splits, int axis);

// Split a tensor at user-specified indices along an axis.
//
// Produces ``indices.size() + 1`` pieces whose sizes along ``axis`` are the
// consecutive differences between ``0``, the sorted ``indices``, and
// ``a.shape[axis]``.  Each piece carries its own ``SplitSliceBackward`` node.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// indices : vector<int64_t>
//     Strictly increasing split positions in the range
//     ``[0, a.shape[axis]]``.
// axis : int
//     Axis to split along.  Supports negative indexing.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``indices.size() + 1`` tensors whose ``axis`` dimensions partition
//     ``a.shape[axis]``.
//
// See Also
// --------
// :func:`split_op`.
LUCID_API std::vector<TensorImplPtr>
split_at_op(const TensorImplPtr& a, std::vector<std::int64_t> indices, int axis);

// Split a tensor into a fixed number of equal chunks along an axis.
//
// Alias for :func:`split_op` with ``num_splits = chunks``.  Provided for
// API symmetry with reference framework's ``chunk``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// chunks : int64_t
//     Number of chunks; must divide ``a.shape[axis]`` exactly.
// axis : int
//     Axis to chunk along.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``chunks`` equal-sized tensors.
//
// See Also
// --------
// :func:`split_op`.
LUCID_API std::vector<TensorImplPtr>
chunk_op(const TensorImplPtr& a, std::int64_t chunks, int axis);

// Unbind a tensor into a list of slices along an axis, dropping that axis.
//
// Produces ``a.shape[axis]`` tensors, each with rank ``ndim - 1``.  The
// k-th output is ``a[..., k, ...]`` along ``axis`` with the split dimension
// squeezed away.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor with rank >= 1.
// axis : int
//     Axis to unbind along.  Supports negative indexing.
//
// Returns
// -------
// vector<TensorImplPtr>
//     ``a.shape[axis]`` tensors of rank ``ndim - 1``.
//
// Notes
// -----
// Backward inserts the gradient slice back at offset ``k`` along ``axis``
// (after unsqueezing) before scattering into the input gradient.
//
// See Also
// --------
// :func:`stack_op` (the inverse), :func:`split_op`.
LUCID_API std::vector<TensorImplPtr> unbind_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
