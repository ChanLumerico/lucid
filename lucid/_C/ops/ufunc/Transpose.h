#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

class LUCID_API PermuteBackward : public FuncOp<PermuteBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<int> perm_;

    static TensorImplPtr forward(const TensorImplPtr& a, const std::vector<int>& perm);

    std::vector<Storage> apply(Storage grad_out) override;
};

LUCID_API TensorImplPtr permute_op(const TensorImplPtr& a, const std::vector<int>& perm);

LUCID_API TensorImplPtr transpose_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr T_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr mT_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr swapaxes_op(const TensorImplPtr& a, int axis1, int axis2);

}  // namespace lucid
