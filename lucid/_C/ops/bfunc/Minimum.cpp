#include "Minimum.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MinimumBackward::schema_v1{"minimum", 1, AmpPolicy::Promote, true};

std::pair<Storage, Storage> MinimumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);

    Storage mask_a = ge_mask_storage(saved_inputs_[1], saved_inputs_[0], n, dtype_, device_);
    Storage mask_b = lt_mask_storage(saved_inputs_[1], saved_inputs_[0], n, dtype_, device_);
    Storage dx = multiply_storages(grad_out, mask_a, n, dtype_, device_);
    Storage dy = multiply_storages(grad_out, mask_b, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

TensorImplPtr minimum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MinimumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MinimumBackward)

}  // namespace lucid
