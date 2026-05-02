#include "Add.h"

#include <utility>

#include <mlx/ops.h>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"
#include "../../core/Shape.h"

namespace lucid {

const OpSchema AddBackward::schema_v1{"add", /*version=*/1, AmpPolicy::Promote,
                                      /*deterministic=*/true};

std::pair<Storage, Storage> AddBackward::grad_formula(const Storage& grad_out) {
    // d(a+b)/da = 1, d(a+b)/db = 1. Both grads are exactly grad_out.
    // We clone so each downstream consumer owns its buffer (engine accumulates
    // into pending grads). reduce_grad_to_shape with same shape returns a
    // clone, so we can just call it with the trivial mapping — but that's
    // overkill. Build a direct clone via a no-op reduce.
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
