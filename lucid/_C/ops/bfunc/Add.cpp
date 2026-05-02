#include "Add.h"

#include <utility>

#include <mlx/ops.h>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../core/Shape.h"

namespace lucid {

const OpSchema AddBackward::schema_v1{"add", 1, AmpPolicy::Promote, true};

std::pair<Storage, Storage> AddBackward::grad_formula(const Storage& grad_out) {
    return {
        reduce_grad_to_shape(grad_out, out_shape_, out_shape_, dtype_, device_),
        reduce_grad_to_shape(grad_out, out_shape_, out_shape_, dtype_, device_),
    };
}

TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return AddBackward::forward(a, b);
}

LUCID_REGISTER_OP(AddBackward)

}  // namespace lucid
