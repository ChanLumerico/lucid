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
    if (t_->dtype_ != Dtype::F32 && t_->dtype_ != Dtype::F64) {
        ErrorBuilder(label_).not_implemented("only F32/F64 supported (got " +
                                             std::string(dtype_name(t_->dtype_)) + ")");
    }
    return *this;
}

Validator& Validator::dtype_eq(Dtype expected) {
    non_null();
    if (t_->dtype_ != expected) {
        throw DtypeMismatch(std::string(dtype_name(expected)), std::string(dtype_name(t_->dtype_)),
                            label_);
    }
    return *this;
}

Validator& Validator::dtype_in(std::initializer_list<Dtype> allowed) {
    non_null();
    for (Dtype d : allowed)
        if (t_->dtype_ == d)
            return *this;
    std::string allowed_str;
    bool first = true;
    for (Dtype d : allowed) {
        if (!first)
            allowed_str += "|";
        allowed_str += dtype_name(d);
        first = false;
    }
    throw DtypeMismatch(allowed_str, std::string(dtype_name(t_->dtype_)), label_);
}

Validator& Validator::ndim(int expected) {
    non_null();
    if (static_cast<int>(t_->shape_.size()) != expected) {
        ErrorBuilder(label_).fail("expected ndim=" + std::to_string(expected) +
                                  ", got ndim=" + std::to_string(t_->shape_.size()));
    }
    return *this;
}

Validator& Validator::ndim_at_least(int min_n) {
    non_null();
    if (static_cast<int>(t_->shape_.size()) < min_n) {
        ErrorBuilder(label_).fail("expected ndim>=" + std::to_string(min_n) +
                                  ", got ndim=" + std::to_string(t_->shape_.size()));
    }
    return *this;
}

Validator& Validator::shape_eq(const Shape& expected) {
    non_null();
    if (t_->shape_ != expected) {
        throw ShapeMismatch(expected, t_->shape_, label_);
    }
    return *this;
}

Validator& Validator::square_2d() {
    non_null();
    if (t_->shape_.size() < 2) {
        ErrorBuilder(label_).fail("expected >=2-D");
    }
    const auto n = t_->shape_.size();
    if (t_->shape_[n - 1] != t_->shape_[n - 2]) {
        ErrorBuilder(label_).fail("last two dims must be equal (square)");
    }
    return *this;
}

// ---------- Pair ---------- //

Validator::Pair::Pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op)
    : a_(a), b_(b), op_(std::move(op)) {}

Validator::Pair Validator::pair(const TensorImplPtr& a,
                                const TensorImplPtr& b,
                                std::string op_name) {
    return Pair(a, b, std::move(op_name));
}

Validator::Pair& Validator::Pair::both_non_null() {
    if (!a_ || !b_)
        ErrorBuilder(op_).fail("null input");
    return *this;
}

Validator::Pair& Validator::Pair::same_dtype() {
    both_non_null();
    if (a_->dtype_ != b_->dtype_) {
        throw DtypeMismatch(std::string(dtype_name(a_->dtype_)),
                            std::string(dtype_name(b_->dtype_)), op_);
    }
    return *this;
}

Validator::Pair& Validator::Pair::same_device() {
    both_non_null();
    if (a_->device_ != b_->device_) {
        throw DeviceMismatch(std::string(device_name(a_->device_)),
                             std::string(device_name(b_->device_)), op_);
    }
    return *this;
}

Validator::Pair& Validator::Pair::same_shape() {
    both_non_null();
    if (a_->shape_ != b_->shape_) {
        throw ShapeMismatch(a_->shape_, b_->shape_, op_);
    }
    return *this;
}

}  // namespace lucid
