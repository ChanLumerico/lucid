#include "Inplace.h"

#include <utility>

#include "../../core/Exceptions.h"
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

// Compute fwd_fn(a, b), then overwrite a's storage with the result,
// bump version, and return a. Shape/dtype/device must match.
template <typename Fn>
TensorImplPtr inplace_apply(const TensorImplPtr& a, const TensorImplPtr& b,
                            Fn&& fwd_fn, const char* name) {
    if (!a || !b) throw LucidError(std::string(name) + ": null input");
    auto out = fwd_fn(a, b);
    if (out->shape_ != a->shape_)
        throw ShapeMismatch(a->shape_, out->shape_,
                            std::string(name) + " (in-place: shape changed)");
    a->storage_ = std::move(out->storage_);
    a->dtype_   = out->dtype_;
    a->device_  = out->device_;
    a->version_ += 1;
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
