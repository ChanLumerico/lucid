#include "Validate.h"

#include <set>

#include "Dtype.h"
#include "Error.h"
#include "ErrorBuilder.h"
#include "TensorImpl.h"

namespace lucid {

Validator Validator::input(const TensorImplPtr& t, std::string label) {
    return Validator(t, std::move(label));
}

Validator& Validator::non_null() {
    if (!t_)
        ErrorBuilder(label_).fail("null input");
    return *this;
}

Validator& Validator::float_only() {
    non_null();
    if (t_->dtype() != Dtype::F32 && t_->dtype() != Dtype::F64) {
        ErrorBuilder(label_).not_implemented("only F32/F64 supported (got " +
                                             std::string(dtype_name(t_->dtype())) + ")");
    }
    return *this;
}

Validator& Validator::dtype_eq(Dtype expected) {
    non_null();
    if (t_->dtype() != expected) {
        throw DtypeMismatch(std::string(dtype_name(expected)), std::string(dtype_name(t_->dtype())),
                            label_);
    }
    return *this;
}

Validator& Validator::dtype_in(std::initializer_list<Dtype> allowed) {
    non_null();
    for (Dtype d : allowed)
        if (t_->dtype() == d)
            return *this;
    std::string allowed_str;
    bool first = true;
    for (Dtype d : allowed) {
        if (!first)
            allowed_str += "|";
        allowed_str += dtype_name(d);
        first = false;
    }
    throw DtypeMismatch(allowed_str, std::string(dtype_name(t_->dtype())), label_);
}

Validator& Validator::ndim(int expected) {
    non_null();
    if (static_cast<int>(t_->shape().size()) != expected) {
        ErrorBuilder(label_).fail("expected ndim=" + std::to_string(expected) +
                                  ", got ndim=" + std::to_string(t_->shape().size()));
    }
    return *this;
}

Validator& Validator::ndim_at_least(int min_n) {
    non_null();
    if (static_cast<int>(t_->shape().size()) < min_n) {
        ErrorBuilder(label_).fail("expected ndim>=" + std::to_string(min_n) +
                                  ", got ndim=" + std::to_string(t_->shape().size()));
    }
    return *this;
}

Validator& Validator::shape_eq(const Shape& expected) {
    non_null();
    if (t_->shape() != expected) {
        throw ShapeMismatch(expected, t_->shape(), label_);
    }
    return *this;
}

Validator& Validator::square_2d() {
    non_null();
    if (t_->shape().size() < 2) {
        ErrorBuilder(label_).fail("expected >=2-D");
    }
    const auto n = t_->shape().size();
    if (t_->shape()[n - 1] != t_->shape()[n - 2]) {
        ErrorBuilder(label_).fail("last two dims must be equal (square)");
    }
    return *this;
}

Validator::Pair::Pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op)
    : a_(a), b_(b), op_(std::move(op)) {}

Validator::Pair
Validator::pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op_name) {
    return Pair(a, b, std::move(op_name));
}

Validator::Pair& Validator::Pair::both_non_null() {
    if (!a_ || !b_)
        ErrorBuilder(op_).fail("null input");
    return *this;
}

Validator::Pair& Validator::Pair::same_dtype() {
    both_non_null();
    if (a_->dtype() != b_->dtype()) {
        throw DtypeMismatch(std::string(dtype_name(a_->dtype())),
                            std::string(dtype_name(b_->dtype())), op_);
    }
    return *this;
}

Validator::Pair& Validator::Pair::same_device() {
    both_non_null();
    if (a_->device() != b_->device()) {
        throw DeviceMismatch(std::string(device_name(a_->device())),
                             std::string(device_name(b_->device())), op_);
    }
    return *this;
}

Validator::Pair& Validator::Pair::same_shape() {
    both_non_null();
    if (a_->shape() != b_->shape()) {
        throw ShapeMismatch(a_->shape(), b_->shape(), op_);
    }
    return *this;
}

}  // namespace lucid
