#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

class LUCID_API BroadcastBackward : public FuncOp<BroadcastBackward, 1> {
public:
    static const OpSchema schema_v1;
    Shape input_shape_;
    Shape output_shape_;
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis);

LUCID_API TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape);

LUCID_API TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape);

}  // namespace lucid
