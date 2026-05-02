#pragma once

#include <initializer_list>
#include <string>

#include "../api.h"
#include "Dtype.h"
#include "Error.h"
#include "Shape.h"
#include "TensorImpl.h"
#include "fwd.h"

namespace lucid {

class LUCID_API Validator {
public:
    static Validator input(const TensorImplPtr& t, std::string label);

    Validator& non_null();
    Validator& float_only();
    Validator& dtype_eq(Dtype expected);
    Validator& dtype_in(std::initializer_list<Dtype> allowed);
    Validator& ndim(int expected);
    Validator& ndim_at_least(int min_n);
    Validator& shape_eq(const Shape& expected);
    Validator& square_2d();

    class LUCID_API Pair {
    public:
        Pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op);
        Pair& both_non_null();
        Pair& same_dtype();
        Pair& same_device();
        Pair& same_shape();

    private:
        const TensorImplPtr& a_;
        const TensorImplPtr& b_;
        std::string op_;
    };
    static Pair pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op_name);

private:
    Validator(const TensorImplPtr& t, std::string label) : t_(t), label_(std::move(label)) {}
    const TensorImplPtr& t_;
    std::string label_;
};

}  // namespace lucid
