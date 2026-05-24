// lucid/_C/ops/utils/View.h
//
// Declares the autograd node and public free functions for shape-reinterpretation
// ops: reshape, squeeze, and unsqueeze.  All three produce a new TensorImpl that
// shares the underlying Storage with the input rather than copying data; they are
// therefore only valid when the input tensor is contiguous (verified by the
// backend dispatcher's reshape implementation).

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for reshape, squeeze, and unsqueeze.
//
// Forward reinterprets the flat element buffer under a new shape.  The
// backend dispatcher's reshape creates a new ``Storage`` alias over the same
// physical buffer; no data movement occurs.  Backward reshapes the incoming
// gradient back to the original input shape recorded in ``input_shapes_[0]``;
// because reshape is its own inverse on contiguous buffers (element ordering
// is unchanged), the backward pass is simply another reshape call.
//
// Parameters
// ----------
// grad_out : Storage
//     Incoming gradient, shaped according to the forward output
//     (``out_shape_``).
//
// Returns
// -------
// vector<Storage>
//     Single-element vector containing the gradient reshaped to
//     ``input_shapes_[0]``.
//
// Shape
// -----
// ``input_shapes_[0]`` — shape of the input tensor before the reshape.
// ``out_shape_``       — shape after the reshape (inherited from
//                        ``FuncOp``).  Both must have identical element
//                        counts.
//
// Math
// ----
// $$ y_k = x_k \quad \text{for } k \in [0, \mathrm{numel}(x)), $$
// where indices are taken in row-major flat ordering.  The mapping between
// multi-dim indices changes, but the linear element ordering does not.
//
// Notes
// -----
// In graph mode ``apply_for_graph`` routes through :func:`reshape_op` so the
// reshape itself becomes a tracked node and the upstream gradient flow
// participates in graph optimisation.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"view"``, ``AmpPolicy::KeepInput``, requires_grad=true, is_view=true.
//
// See Also
// --------
// :func:`reshape_op`, :func:`squeeze_op`, :func:`unsqueeze_op`,
// :func:`contiguous_op`.
class LUCID_API ViewBackward : public FuncOp<ViewBackward, 1> {
public:
    static const OpSchema schema_v1;
    // Reshape grad_out from out_shape_ back to input_shapes_[0].
    std::vector<Storage> apply(Storage grad_out) override;
    // Graph-mode: reshape via reshape_op so the result is tracked.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
    // Graph label — ``"reshape"`` — for debug printing and profiler
    // traces.  Overrides :func:`Node::node_name`.
    std::string node_name() const override { return "reshape"; }
};

// Reinterpret a tensor under a new shape (no data copy).
//
// The total element count must be preserved.  Exactly one entry of
// ``new_shape`` may be ``-1``, in which case that dimension is inferred from
// the remaining sizes.  The input must be contiguous; pass non-contiguous
// inputs through :func:`contiguous_op` first.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  Must be C-contiguous.
// new_shape : vector<int64_t>
//     Target shape.  At most one entry may be ``-1``.  All other entries
//     must be non-negative.  Since ``Shape == vector<int64_t>``, this
//     overload also accepts ``Shape`` directly.
//
// Returns
// -------
// TensorImplPtr
//     A view tensor with shape ``new_shape`` (with ``-1`` resolved) sharing
//     storage with ``a``.
//
// Shape
// -----
// $\mathrm{numel}(a) = \prod_d \text{new\_shape}[d]$, after resolving any
// ``-1`` entry.
//
// Raises
// ------
// ShapeMismatch
//     If element counts do not match, multiple ``-1`` entries are given, or
//     the input is non-contiguous.
//
// Examples
// --------
// ``reshape_op(t_2x3, {6})`` flattens a 2x3 to a 1-D length-6 view.
// ``reshape_op(t_2x6, {-1, 3})`` infers the first dim as ``4``.
//
// See Also
// --------
// :func:`flatten_op`, :func:`squeeze_op`, :func:`contiguous_op`.
LUCID_API TensorImplPtr reshape_op(const TensorImplPtr& a,
                                   const std::vector<std::int64_t>& new_shape);

// Remove a single size-1 dimension.
//
// Drops the axis at position ``dim``, which must currently have size 1.
// Negative ``dim`` wraps relative to the input rank.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor with rank >= 1.
// dim : int
//     Axis to remove.  Must satisfy ``-ndim <= dim < ndim`` and
//     ``a.shape[dim] == 1``.
//
// Returns
// -------
// TensorImplPtr
//     A view with rank ``ndim - 1``.
//
// Raises
// ------
// IndexError
//     If ``dim`` is out of range.
// ValueError
//     If the targeted dimension is not of size 1.
//
// See Also
// --------
// :func:`squeeze_all_op`, :func:`unsqueeze_op`.
LUCID_API TensorImplPtr squeeze_op(const TensorImplPtr& a, int dim);

// Remove every size-1 dimension from a tensor.
//
// Equivalent to repeated application of :func:`squeeze_op` on each size-1
// axis.  If no such axes exist, returns a view with the same shape.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     A view with all size-1 dimensions removed.  Rank may equal the
//     input's rank if no size-1 axes were present.
//
// See Also
// --------
// :func:`squeeze_op`.
LUCID_API TensorImplPtr squeeze_all_op(const TensorImplPtr& a);

// Insert a new size-1 dimension at a given position.
//
// The new axis is added before position ``dim`` in the output, so the output
// rank is ``ndim + 1``.  Negative ``dim`` wraps relative to the output rank
// (so ``dim = -1`` inserts before the last existing dimension).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// dim : int
//     Insertion position.  Must satisfy ``-(ndim + 1) <= dim <= ndim``.
//
// Returns
// -------
// TensorImplPtr
//     A view with rank ``ndim + 1`` and a size-1 dimension inserted at
//     ``dim``.
//
// Raises
// ------
// IndexError
//     If ``dim`` is out of range.
//
// Examples
// --------
// A length-5 1-D tensor unsqueezed at ``dim=0`` becomes a ``(1, 5)`` view;
// unsqueezed at ``dim=1`` becomes a ``(5, 1)`` view.
//
// See Also
// --------
// :func:`squeeze_op`, :func:`reshape_op`.
LUCID_API TensorImplPtr unsqueeze_op(const TensorImplPtr& a, int dim);

}  // namespace lucid
