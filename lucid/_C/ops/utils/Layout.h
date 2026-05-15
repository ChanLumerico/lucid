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

// Autograd node for broadcast_to (and its alias expand_op).
//
// Forward:  replicate `a` along all broadcast dimensions so that its shape
//           matches the requested output shape, following NumPy broadcasting
//           rules (left-pad with 1s, then allow size-1 dims to broadcast).
// Backward: sum the incoming gradient over every axis that was broadcast
//           (either a newly prepended axis or a size-1 dimension that was
//           expanded).  This is performed by Dispatcher::reduce_broadcast.
//
// Invariants:
//   input_shape_  — the original shape before broadcasting.
//   output_shape_ — the fully broadcast target shape.
class LUCID_API BroadcastBackward : public FuncOp<BroadcastBackward, 1> {
public:
    static const OpSchema schema_v1;
    Shape input_shape_;
    Shape output_shape_;
    std::vector<Storage> apply(Storage grad_out) override;
};

// Collapse a contiguous range of axes [start_axis, end_axis] (inclusive) into
// a single dimension.  The axes outside this range are left unchanged.
// Delegates to reshape_op after computing the flattened size.
LUCID_API TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis);

// Broadcast `a` to `shape` following NumPy semantics.  Raises ShapeMismatch
// if the input cannot be broadcast to the requested shape.  If `a` is not
// already contiguous a copy is made before the broadcast.
LUCID_API TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape);

// Alias for broadcast_to_op.  Provided for API symmetry with reference framework's
// Tensor::expand.
LUCID_API TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape);

}  // namespace lucid
