// lucid/_C/ops/ufunc/Trig.h
//
// Autograd backward nodes and entry points for trigonometric operations:
// sin, cos, tan, arcsin, arccos, arctan.  On CPU, the backend delegates to
// vForce (vvsinf, vvcosf, …) for vectorised throughput.  All ops use
// AmpPolicy::Promote so that float64 inputs retain their precision.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Backward node for element-wise sine: y = sin(x).
//
// Gradient rule: dL/dx = cos(x) * dL/dy.
// Saves the input so grad_formula can evaluate cos(x).
class LUCID_API SinBackward : public UnaryOp<SinBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sin(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr sin_op(const TensorImplPtr& a);

// Backward node for element-wise cosine: y = cos(x).
//
// Gradient rule: dL/dx = -sin(x) * dL/dy.
// Saves the input; grad_formula computes sin(x) and negates it.
class LUCID_API CosBackward : public UnaryOp<CosBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cos(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr cos_op(const TensorImplPtr& a);

// Backward node for element-wise tangent: y = tan(x).
//
// Gradient rule: dL/dx = dL/dy / cos^2(x).
// Saves the input; grad_formula squares cos(x) and divides.
class LUCID_API TanBackward : public UnaryOp<TanBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.tan(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr tan_op(const TensorImplPtr& a);

// Backward node for element-wise arcsine: y = arcsin(x), domain x in (-1, 1).
//
// Gradient rule: dL/dx = dL/dy / sqrt(1 - x^2).
// Saves the input; grad_formula builds the radicand 1 - x^2 from saved_inputs_.
class LUCID_API AsinBackward : public UnaryOp<AsinBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.asin(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr arcsin_op(const TensorImplPtr& a);

// Backward node for element-wise arccosine: y = arccos(x), domain x in (-1, 1).
//
// Gradient rule: dL/dx = -dL/dy / sqrt(1 - x^2).
// Same radicand as arcsin, but negated.
class LUCID_API AcosBackward : public UnaryOp<AcosBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.acos(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr arccos_op(const TensorImplPtr& a);

// Backward node for element-wise arctangent: y = arctan(x).
//
// Gradient rule: dL/dx = dL/dy / (1 + x^2).
// Saves the input; grad_formula computes 1 + x^2 as the denominator.
class LUCID_API AtanBackward : public UnaryOp<AtanBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.atan(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr arctan_op(const TensorImplPtr& a);

}  // namespace lucid
