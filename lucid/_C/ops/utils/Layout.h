#pragma once

// =====================================================================
// Layout transforms: pure shape changes that may also broadcast values.
//   flatten / broadcast_to / expand
// =====================================================================

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Backward for broadcast_to / expand: sum the gradient along the broadcast
// dimensions back to the original input shape.
/// Autograd backward node for Broadcast.
class LUCID_API BroadcastBackward : public FuncOp<BroadcastBackward, 1> {
public:
    static const OpSchema schema_v1;
    Shape input_shape_;
    Shape output_shape_;
    std::vector<Storage> apply(Storage grad_out) override;
};

/// Flatten.
LUCID_API TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis);
/// Broadcast to.
LUCID_API TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape);
/// Expand.
LUCID_API TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape);

}  // namespace lucid
