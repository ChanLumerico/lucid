#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

class LUCID_API ViewBackward : public FuncOp<ViewBackward, 1> {
public:
    static const OpSchema schema_v1;
    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr reshape_op(const TensorImplPtr& a,
                                   const std::vector<std::int64_t>& new_shape);

LUCID_API TensorImplPtr squeeze_op(const TensorImplPtr& a, int dim);

LUCID_API TensorImplPtr squeeze_all_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr unsqueeze_op(const TensorImplPtr& a, int dim);

}  // namespace lucid
