// lucid/_C/ops/ufunc/ScalarParam.h
//
// Backward nodes for unary ops that carry a scalar hyper-parameter:
//   - PowScalarBackward  — x^exp  (exp is a floating-point scalar exponent)
//   - RPowScalarBackward — base^x (base is a floating-point scalar base)
//   - ClipBackward       — clamp(x, min, max)
//
// All three override the standard static forward() from UnaryKernel so they
// can capture the scalar on the backward node.  They do not use the generic
// dispatch() path.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Backward node for scalar-exponent power: y = x^exp.
//
// Gradient rule: dL/dx = exp * x^(exp-1) * dL/dy.
// exp_ is persisted on the node; forward() uses it to compute x^(exp-1)
// during the backward call through the backend.
class LUCID_API PowScalarBackward : public UnaryOp<PowScalarBackward> {
public:
    double exp_ = 0.0;
    static const OpSchema schema_v1;
    // Override: captures exp and dispatches to backend::pow_scalar.
    static TensorImplPtr forward(const TensorImplPtr& a, double exp);
    Storage grad_formula(const Storage& g);
};

// Backward node for scalar-base reverse power: y = base^x.
//
// Gradient rule: dL/dx = ln(base) * y * dL/dy.
// Saves the *output* y (= base^x) because the formula uses y directly;
// base_ is persisted to compute ln(base) in grad_formula.
class LUCID_API RPowScalarBackward : public UnaryOp<RPowScalarBackward> {
public:
    double base_ = 0.0;

    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Override: captures base, manually saves the output, and wires autograd.
    static TensorImplPtr forward(double base, const TensorImplPtr& a);
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise clip (clamp): y = clamp(x, min, max).
//
// Gradient rule: dL/dx = 1 if min < x < max, else 0.
// min_ and max_ are persisted so grad_formula can reconstruct the in-range mask.
class LUCID_API ClipBackward : public UnaryOp<ClipBackward> {
public:
    double min_ = 0.0;
    double max_ = 0.0;
    static const OpSchema schema_v1;
    // Override: captures min_v, max_v and wires the backward node manually.
    static TensorImplPtr forward(const TensorImplPtr& a, double min_v, double max_v);
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr pow_scalar_op(const TensorImplPtr& a, double exp);

LUCID_API TensorImplPtr rpow_scalar_op(double base, const TensorImplPtr& a);

LUCID_API TensorImplPtr clip_op(const TensorImplPtr& a, double min_v, double max_v);

}  // namespace lucid
