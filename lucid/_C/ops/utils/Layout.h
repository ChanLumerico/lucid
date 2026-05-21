// lucid/_C/ops/utils/Layout.h
//
// Declares layout-manipulation ops: flatten, broadcast_to, and expand.
// These reshape or replicate tensor data along dimensions without changing the
// underlying values; broadcast_to and expand carry a backward pass that
// reduces the gradient back to the original shape by summing over broadcast
// dimensions.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for broadcasting a tensor to a larger shape.
//
// Forward replicates ``a`` along all broadcast dimensions so that its shape
// matches the requested output shape, following NumPy broadcasting rules
// (left-pad with size-1 dims, then allow size-1 dims to broadcast).  Backward
// inverts that replication by summing the incoming gradient over every axis
// that was broadcast — either a newly prepended axis or a size-1 dimension
// that was expanded — via ``Dispatcher::reduce_broadcast``.
//
// Parameters
// ----------
// input_shape_ : Shape
//     The original tensor shape before broadcasting.
// output_shape_ : Shape
//     The fully broadcast target shape.
//
// Math
// ----
// Forward expands each size-1 broadcast axis $k$ by replication:
// $$ y_{[\ldots, i_k, \ldots]} = x_{[\ldots, 0, \ldots]} \quad \text{for all } i_k. $$
// Backward reduces over those axes:
// $$ \frac{\partial L}{\partial x_{[\ldots, 0, \ldots]}}
//        = \sum_{i_k} \frac{\partial L}{\partial y_{[\ldots, i_k, \ldots]}}. $$
//
// Shape
// -----
// Output rank == ``output_shape_.size()``.  For each axis from the right,
// either ``input_shape_[d] == output_shape_[d]`` or
// ``input_shape_[d] == 1`` (broadcast).  Newly prepended axes are
// implicitly summed in backward.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"broadcast_to"``, ``AmpPolicy::KeepInput``, requires_grad=true.
//
// See Also
// --------
// :func:`broadcast_to_op`, :func:`expand_op`.
class LUCID_API BroadcastBackward : public FuncOp<BroadcastBackward, 1> {
public:
    static const OpSchema schema_v1;
    Shape input_shape_;
    Shape output_shape_;
    std::vector<Storage> apply(Storage grad_out) override;
};

// Collapse a contiguous range of axes into a single dimension.
//
// Folds dimensions ``[start_axis, end_axis]`` (inclusive on both ends) into
// one axis whose size is the product of the collapsed dimensions.  Delegates
// to :func:`reshape_op` after computing the flattened size, so the input
// must be contiguous along the affected range.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// start_axis : int
//     First axis in the range to collapse.  Supports negative indexing.
// end_axis : int
//     Last axis in the range (inclusive).  Supports negative indexing.
//     Must satisfy ``start_axis <= end_axis`` after normalisation.
//
// Returns
// -------
// TensorImplPtr
//     Tensor with rank
//     ``ndim - (end_axis - start_axis)`` and a flattened dimension of size
//     ``\prod_{d=start_axis}^{end_axis} a.shape[d]`` at position ``start_axis``.
//
// Shape
// -----
// ``(D_0, ..., D_{s-1}, D_s, ..., D_e, D_{e+1}, ..., D_{n-1})`` becomes
// ``(D_0, ..., D_{s-1}, \prod_{d=s}^{e} D_d, D_{e+1}, ..., D_{n-1})``
// with ``s = start_axis`` and ``e = end_axis``.
//
// Raises
// ------
// ShapeMismatch
//     If the collapsed range cannot be reshape-merged because of non-contiguous
//     strides.
//
// See Also
// --------
// :func:`reshape_op`, :func:`squeeze_all_op`.
LUCID_API TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis);

// Broadcast a tensor to a larger shape following NumPy rules.
//
// The input shape is right-aligned with ``shape``; size-1 dimensions are
// replicated and newly prepended axes are inserted.  If ``a`` is not
// contiguous a copy is made before the broadcast operation runs.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// shape : Shape
//     Target shape.  Must be NumPy-broadcast-compatible with ``a.shape``.
//
// Returns
// -------
// TensorImplPtr
//     A tensor of shape ``shape`` whose values are replicated from ``a``.
//
// Raises
// ------
// ShapeMismatch
//     If ``a.shape`` cannot be broadcast to ``shape`` (any mismatching axis
//     where neither side is 1).
//
// Examples
// --------
// ``broadcast_to_op([1, 2, 3], (4, 3))`` produces a 4x3 tensor where every
// row equals ``[1, 2, 3]``.
//
// See Also
// --------
// :func:`expand_op` (alias), :func:`BroadcastBackward`.
LUCID_API TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape);

// Alias for :func:`broadcast_to_op` with reference framework's expand semantics.
//
// Provided so callers porting from reference framework can keep the call site
// unchanged.  Behaviour is identical to :func:`broadcast_to_op`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// shape : Shape
//     Target shape (NumPy-broadcast-compatible with ``a.shape``).
//
// Returns
// -------
// TensorImplPtr
//     Broadcast tensor.
//
// See Also
// --------
// :func:`broadcast_to_op`.
LUCID_API TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape);

}  // namespace lucid
