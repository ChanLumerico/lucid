#include "Inplace.h"

#include <utility>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "Add.h"
#include "Div.h"
#include "Maximum.h"
#include "Minimum.h"
#include "Mul.h"
#include "Pow.h"
#include "Sub.h"

namespace lucid {

namespace {

template <typename Fn>
TensorImplPtr
inplace_apply(const TensorImplPtr& a, const TensorImplPtr& b, Fn&& fwd_fn, const char* name) {
    if (!a || !b)
        ErrorBuilder(name).fail("null input");
    if (a->storage_is_shared())
        ErrorBuilder(name).fail("in-place op on a tensor that shares storage with a view — "
                                "call .clone() first or operate on the base tensor");
    auto out = fwd_fn(a, b);
    if (out->shape() != a->shape())
        throw ShapeMismatch(a->shape(), out->shape(),
                            std::string(name) + " (in-place: shape changed)");
    a->mutable_storage() = std::move(out->mutable_storage());
    a->set_dtype(out->dtype());
    a->set_device(out->device());
    a->bump_version();
    return a;
}

}  // namespace

TensorImplPtr add_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &add_op, "add_");
}
TensorImplPtr sub_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &sub_op, "sub_");
}
TensorImplPtr mul_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &mul_op, "mul_");
}
TensorImplPtr div_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &div_op, "div_");
}
TensorImplPtr pow_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &pow_op, "pow_");
}
TensorImplPtr maximum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &maximum_op, "maximum_");
}
TensorImplPtr minimum_inplace_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return inplace_apply(a, b, &minimum_op, "minimum_");
}

}  // namespace lucid
