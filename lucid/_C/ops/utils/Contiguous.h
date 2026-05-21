// lucid/_C/ops/utils/Contiguous.h
//
// Declares the autograd node and public entry point for the contiguous op.
// The op materialises a densely-packed, row-major (C-contiguous) copy of a
// tensor that may have non-unit strides, a non-zero storage offset, or a
// transposed layout.  If the input is already contiguous the backend may
// return a view rather than a copy; the autograd node handles both cases.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for materialising a C-contiguous copy of a tensor.
//
// Forward calls ``Dispatcher::contiguous``, which copies non-contiguous data
// into a freshly allocated, densely laid-out buffer.  If the input is already
// contiguous the backend may return the same storage without performing a
// copy; the decision uses ``is_contiguous`` together with stride and offset
// metadata.  Backward is essentially pass-through: because gradients are
// always dense, ``apply`` simply clones the gradient storage so upstream
// nodes receive a concrete, owning buffer with the input shape and dtype.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor with arbitrary strides, offset, and layout.
//
// Returns
// -------
// TensorImplPtr
//     Output sharing the input's shape and dtype, with canonical row-major
//     strides and storage offset zero.
//
// Shape
// -----
// ``out.shape == in.shape`` exactly; only the stride and offset metadata
// (and the underlying physical buffer) may change.
//
// Math
// ----
// $$ y_{[i_0, \ldots, i_{n-1}]} = x_{[i_0, \ldots, i_{n-1}]} $$
// (identity on elements; the operation differs only in physical layout).
//
// Notes
// -----
// Required as a fast-path normalisation before any op that assumes
// row-major storage (notably :func:`reshape_op` and most MLX kernels).
// The backward pass clones the dense gradient because the original input
// could have had any layout — cloning guarantees the upstream gradient is
// a concrete buffer regardless of the forward path taken.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"contiguous"``, ``AmpPolicy::KeepInput``, requires_grad=true.
//
// See Also
// --------
// :func:`reshape_op` — typically immediately follows ``contiguous_op``.
class LUCID_API ContiguousBackward : public FuncOp<ContiguousBackward, 1> {
public:
    static const OpSchema schema_v1;
    // Run the forward pass and wire autograd in one step.  Passes stride,
    // storage_offset, and is_contiguous metadata to the backend dispatcher.
    static TensorImplPtr forward(const TensorImplPtr& a);
    // Clone the dense gradient storage so that upstream nodes receive a
    // concrete buffer regardless of the original input's layout.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Materialise a C-contiguous copy of a tensor.
//
// Delegates to :func:`ContiguousBackward::forward`, which handles both the
// copy and the autograd node attachment.  If ``a`` is already contiguous
// the backend may return ``a`` itself (no copy); otherwise a fresh buffer
// is allocated.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (any layout).
//
// Returns
// -------
// TensorImplPtr
//     A tensor with the same values, shape, and dtype as ``a`` but with
//     canonical row-major strides and ``storage_offset == 0``.
//
// Notes
// -----
// Idempotent: ``contiguous_op(contiguous_op(a))`` is equivalent to
// ``contiguous_op(a)`` and avoids a second copy whenever possible.
//
// See Also
// --------
// :func:`reshape_op`, :func:`broadcast_to_op`.
LUCID_API TensorImplPtr contiguous_op(const TensorImplPtr& a);

}  // namespace lucid
