// lucid/_C/core/Validate.h
//
// Fluent validation API for TensorImpl inputs.  Op implementations call
// Validator::input() or Validator::pair() to build a chain of precondition
// checks; each check throws the appropriate typed exception on failure and
// returns *this on success so checks can be chained.
//
// Example (single tensor):
//   Validator::input(t, "linear.weight")
//       .non_null()
//       .float_only()
//       .ndim(2);
//
// Example (two tensors):
//   Validator::pair(a, b, "matmul")
//       .both_non_null()
//       .same_dtype()
//       .same_device();
//
// The label / op_name argument is passed to ErrorBuilder so that exception
// messages identify the problematic tensor by name.

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

// Single-tensor validator.  Constructed via the static input() factory;
// not intended to be instantiated directly.
class LUCID_API Validator {
public:
    // Creates a Validator for tensor t, using label in error messages.
    static Validator input(const TensorImplPtr& t, std::string label);

    // Throws LucidError if t is null.
    Validator& non_null();

    // Throws NotImplementedError if dtype is not F32 or F64.
    // Calls non_null() internally.
    Validator& float_only();

    // Throws DtypeMismatch if dtype != expected.
    Validator& dtype_eq(Dtype expected);

    // Throws DtypeMismatch if dtype is not in allowed.
    Validator& dtype_in(std::initializer_list<Dtype> allowed);

    // Throws LucidError if ndim != expected.
    Validator& ndim(int expected);

    // Throws LucidError if ndim < min_n.
    Validator& ndim_at_least(int min_n);

    // Throws ShapeMismatch if shape != expected.
    Validator& shape_eq(const Shape& expected);

    // Throws LucidError if the tensor is not at least 2-D, or if its last two
    // dimensions are not equal (i.e. not square in the trailing axes).
    Validator& square_2d();

    // Two-tensor validator.  Constructed via Validator::pair().
    class LUCID_API Pair {
    public:
        Pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op);
        // Throws LucidError if either tensor is null.
        Pair& both_non_null();
        // Throws DtypeMismatch if a.dtype != b.dtype.
        Pair& same_dtype();
        // Throws DeviceMismatch if a.device != b.device.
        Pair& same_device();
        // Throws ShapeMismatch if a.shape != b.shape.
        Pair& same_shape();

    private:
        const TensorImplPtr& a_;
        const TensorImplPtr& b_;
        std::string op_;
    };

    // Creates a Pair validator for tensors a and b, using op_name in messages.
    static Pair pair(const TensorImplPtr& a, const TensorImplPtr& b, std::string op_name);

private:
    Validator(const TensorImplPtr& t, std::string label) : t_(t), label_(std::move(label)) {}
    const TensorImplPtr& t_;
    std::string label_;
};

}  // namespace lucid
