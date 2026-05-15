// lucid/_C/ops/ufunc/Arith.h
//
// Autograd backward nodes and public entry points for the six basic arithmetic
// unary operations: neg, abs, sign, reciprocal, square, cube.  Each class
// follows the standard UnaryOp<Derived> CRTP pattern: a static dispatch()
// routes the forward computation through IBackend, and grad_formula()
// implements the analytic gradient rule.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Backward node for element-wise negation: y = -x.
//
// Gradient rule: dL/dx = -dL/dy (negate the upstream gradient).
// kSavesInput = false because the backward pass requires no saved value.
class LUCID_API NegBackward : public UnaryOp<NegBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.neg(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr&);
};

// Backward node for element-wise absolute value: y = |x|.
//
// Gradient rule: dL/dx = sign(x) * dL/dy.
// Saves the input so that grad_formula can compute sign(x) during the backward
// pass.
class LUCID_API AbsBackward : public UnaryOp<AbsBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.abs(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise sign: y = sign(x).
//
// Gradient rule: dL/dx = 0 everywhere (sign is piecewise constant).
// kHasGradient = false tells UnaryKernel::forward to skip autograd wiring
// entirely; grad_formula returns an empty CpuStorage as a no-op sentinel.
class LUCID_API SignBackward : public UnaryOp<SignBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sign(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise reciprocal: y = 1/x.
//
// Gradient rule: dL/dx = -dL/dy / x^2.
// Saves the input so that grad_formula can compute x^2.
class LUCID_API ReciprocalBackward : public UnaryOp<ReciprocalBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.reciprocal(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
    // dx = -g / x^2
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Backward node for element-wise square: y = x^2.
//
// Gradient rule: dL/dx = 2*x * dL/dy.
// Saves the input to evaluate 2*x in grad_formula.
class LUCID_API SquareBackward : public UnaryOp<SquareBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.square(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
    // dx = 2*x * g
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Backward node for element-wise cube: y = x^3.
//
// Gradient rule: dL/dx = 3*x^2 * dL/dy.
// Saves the input to evaluate 3*x^2 in grad_formula.
class LUCID_API CubeBackward : public UnaryOp<CubeBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cube(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Public entry points.  Each thin wrapper delegates to the corresponding
// backward node's static forward() method, which handles dispatch and autograd.

LUCID_API TensorImplPtr neg_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr abs_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr sign_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr reciprocal_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr square_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr cube_op(const TensorImplPtr& a);

}  // namespace lucid
