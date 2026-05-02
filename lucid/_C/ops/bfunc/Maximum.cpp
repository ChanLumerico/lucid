#include "Maximum.h"

#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

const OpSchema MaximumBackward::schema_v1{"maximum", /*version=*/1, AmpPolicy::Promote,
                                          /*deterministic=*/true};

std::pair<Storage, Storage> MaximumBackward::grad_formula(const Storage& grad_out) {
    const std::size_t n = shape_numel(out_shape_);
    Storage mask_a = ge_mask_storage(saved_inputs_[0], saved_inputs_[1], n, dtype_,
                                     device_);  // a >= b (ties to a)
    Storage mask_b =
        lt_mask_storage(saved_inputs_[0], saved_inputs_[1], n, dtype_, device_);  // a < b
    Storage dx = multiply_storages(grad_out, mask_a, n, dtype_, device_);
    Storage dy = multiply_storages(grad_out, mask_b, n, dtype_, device_);
    return {std::move(dx), std::move(dy)};
}

TensorImplPtr maximum_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MaximumBackward::forward(a, b);
}

LUCID_REGISTER_OP(MaximumBackward)

}  // namespace lucid
