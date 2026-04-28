#pragma once

// =====================================================================
// Lucid C++ engine — fluent input-validation primitives.
// =====================================================================
//
// Replaces ~30 hand-rolled `if (!a) throw…; if (a->dtype() != …) throw…;`
// chains across op forwards with a single chainable expression.
//
// Usage:
//
//   void some_op(const TensorImplPtr& a, const TensorImplPtr& b) {
//       Validator::input(a, "matmul.a")
//           .non_null()
//           .float_only()
//           .ndim_at_least(2);
//       Validator::pair(a, b, "matmul")
//           .same_dtype()
//           .same_device();
//   }
//
// Each method returns `*this` so the chain reads as a sentence. On the first
// failed check the function throws the matching typed exception (Shape /
// Dtype / Device mismatch) and aborts — no silent skips.
//
// Layer: core/. Depends on core/Error.h, core/TensorImpl.h.

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
    /// Start a single-input validation chain. `label` is used in error
    /// messages so users can tell which arg failed.
    static Validator input(const TensorImplPtr& t, std::string label);

    Validator& non_null();
    Validator& float_only();  // dtype in {F32, F64}
    Validator& dtype_eq(Dtype expected);
    Validator& dtype_in(std::initializer_list<Dtype> allowed);
    Validator& ndim(int expected);
    Validator& ndim_at_least(int min_n);
    Validator& shape_eq(const Shape& expected);
    Validator& square_2d();  // last 2 dims must be equal

    // ----- pair-input convenience --------------------------------------- //

    /// Start a paired validation. Returns a small RAII helper that asserts
    /// shape/dtype/device compatibility between two tensors.
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
