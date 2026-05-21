// lucid/_C/ops/utils/Select.h
//
// Declares element-selection and index-based ops: where, masked_fill, roll,
// gather, diagonal, flip, and masked_select.  Several are differentiable;
// their backward passes are implemented in Select.cpp.

#pragma once

#include <cstdint>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Element-wise conditional selection between two tensors.
//
// Returns ``x`` at positions where ``cond`` is true, otherwise ``y``.
// This is the tensor analogue of the ternary expression
// ``cond ? x : y`` applied element-wise.
//
// Parameters
// ----------
// cond : TensorImplPtr
//     Boolean (or 0/1-valued) condition tensor.
// x : TensorImplPtr
//     Values selected at positions where ``cond`` is true.
// y : TensorImplPtr
//     Values selected at positions where ``cond`` is false.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the broadcast shape and dtype of ``x``/``y``.
//
// Math
// ----
// $$\mathrm{out}_i = \begin{cases} x_i & \text{if } \mathrm{cond}_i \\
// y_i & \text{otherwise}\end{cases}$$
//
// Shape
// -----
// On the GPU stream all three operands are broadcast by the MLX backend.
// On the CPU stream all three must share the same shape.
//
// Notes
// -----
// Backward propagates ``grad_out`` masked by ``cond`` to ``x`` and masked
// by ``~cond`` to ``y``.  ``cond`` itself receives no gradient.
LUCID_API TensorImplPtr where_op(const TensorImplPtr& cond,
                                 const TensorImplPtr& x,
                                 const TensorImplPtr& y);

// Replace masked positions with a scalar value.
//
// Returns a tensor equal to ``a`` except at positions where ``mask`` is
// true, which are filled with ``value``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// mask : TensorImplPtr
//     Boolean mask of the same shape as ``a``.
// value : double
//     Fill value (cast to the dtype of ``a``).
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a``.
//
// Notes
// -----
// Backward routes ``grad_out`` through positions where ``mask`` is false
// (the values that survived the fill); positions where ``mask`` is true
// receive zero gradient.  Neither ``mask`` nor ``value`` is differentiable.
LUCID_API TensorImplPtr masked_fill_op(const TensorImplPtr& a,
                                       const TensorImplPtr& mask,
                                       double value);

// Circularly shift elements along one or more axes.
//
// For each ``(shift, axis)`` pair, rotates the tensor along that axis by
// ``shift`` positions; elements that fall off one end wrap around to the
// other.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// shifts : vector<int64>
//     Signed shift amount per axis.
// axes : vector<int>
//     Axes to shift.  Must match ``shifts`` in length.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a``.
//
// Notes
// -----
// Backward applies the inverse circular shift (negated ``shifts``) to the
// incoming gradient, which exactly undoes the forward rotation.
LUCID_API TensorImplPtr roll_op(const TensorImplPtr& a,
                                std::vector<std::int64_t> shifts,
                                std::vector<int> axes);

// Gather elements from ``a`` at index positions along a single axis.
//
// For each location in the output, looks up the value of ``a`` at the
// index given by ``indices`` along ``axis``, holding all other axes fixed.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor.
// indices : TensorImplPtr
//     Integer index tensor of the same rank as ``a``.  Along every axis
//     other than ``axis``, the shape must match ``a``.
// axis : int
//     Axis along which to gather.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``indices.shape`` and dtype of ``a``.
//
// Notes
// -----
// Backward is a scatter-add: the incoming gradient is added back into a
// zero tensor of the input shape at the positions specified by
// ``indices``.  Duplicate indices accumulate, matching NumPy / reference
// framework semantics.
LUCID_API TensorImplPtr gather_op(const TensorImplPtr& a, const TensorImplPtr& indices, int axis);

// Extract a single diagonal as a final-axis vector.
//
// Returns the elements of ``a`` lying on the diagonal of the
// ``(axis1, axis2)`` plane offset by ``offset``: positive offsets shift
// the diagonal toward the upper-right, negative offsets toward the
// lower-left.  The two selected axes are removed and a new trailing axis
// of length ``L`` is appended.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of rank $\ge 2$.
// offset : int
//     Diagonal offset (0 = main diagonal).
// axis1 : int
//     First axis defining the plane.
// axis2 : int
//     Second axis defining the plane.  Must differ from ``axis1``; the
//     two are canonicalised to ``a1 < a2``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor whose shape is ``a.shape`` with axes ``axis1`` and ``axis2``
//     removed and a trailing dimension of length
//     $L = \max(0, \min(M - r_0, N - c_0))$ appended, where $M, N$ are
//     the sizes of the two selected axes and $r_0, c_0$ are the starting
//     row / column for the given offset.
//
// Notes
// -----
// Backward scatters the diagonal gradient into a zero tensor of the
// original input shape, leaving off-diagonal positions zero.
LUCID_API TensorImplPtr diagonal_op(const TensorImplPtr& a, int offset, int axis1, int axis2);

// Reverse a tensor along one or more axes.
//
// Equivalent to NumPy's ``np.flip``: along each listed axis the index
// ordering is inverted.  Unlike :func:`roll_op` there is no wrap-around.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// dims : vector<int>
//     Axes to flip.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of the same shape and dtype as ``a`` with the requested
//     axes reversed.
//
// Notes
// -----
// Backward applies ``flip`` with the same ``dims`` to the incoming
// gradient — flipping is self-inverse.
LUCID_API TensorImplPtr flip_op(const TensorImplPtr& a, std::vector<int> dims);

// Extract elements where a boolean mask is true.
//
// Returns a 1-D tensor containing the elements of ``a`` at the positions
// where ``mask`` is true, in row-major (C) order.  The output length is
// data-dependent.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Source tensor.
// mask : TensorImplPtr
//     Boolean tensor broadcastable to ``a``.
//
// Returns
// -------
// TensorImplPtr
//     1-D tensor of length equal to the number of true entries in
//     ``mask``; dtype matches ``a``.
//
// Notes
// -----
// Because the output size depends on the values of ``mask``, this op
// performs a CPU round-trip on the GPU stream to materialise the count
// before allocation.  Non-differentiable in ``mask``; the gradient
// w.r.t. ``a`` is a scatter into a zero tensor at the selected
// positions.
LUCID_API TensorImplPtr masked_select_op(const TensorImplPtr& a, const TensorImplPtr& mask);

}  // namespace lucid
