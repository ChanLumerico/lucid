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

// Autograd node for the contiguous op.
//
// Forward:  calls Dispatcher::contiguous, which copies non-contiguous data
//           into a freshly allocated, densely-laid-out buffer.  If the input
//           is already contiguous the backend may return the same storage
//           without performing a copy.  The decision is based on the
//           is_contiguous flag together with stride and offset metadata.
// Backward: the gradient arrives at the output shape and must propagate back
//           to the input, which could have had a non-standard layout.
//           Because the gradient itself is always dense, the backward simply
//           clones the gradient storage (preserving shape and dtype) so the
//           upstream node receives a concrete, owning buffer.
//
// Invariants:
//   out_shape_       — the shape of the contiguous output (== input shape).
//   input_shapes_[0] — the shape of the original (possibly non-contiguous) input.
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

// Return a contiguous copy of `a`, or `a` itself if it is already contiguous.
// Delegates to ContiguousBackward::forward, which handles both the copy and
// the autograd node attachment.
LUCID_API TensorImplPtr contiguous_op(const TensorImplPtr& a);

}  // namespace lucid
