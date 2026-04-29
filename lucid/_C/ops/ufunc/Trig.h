#pragma once

// =====================================================================
// Lucid C++ engine — trigonometric unary ops.
// =====================================================================
//
//   sin(x)    grad: cos(x) * g
//   cos(x)    grad: -sin(x) * g
//   tan(x)    grad: g / cos²(x)
//   asin(x)   grad: g / √(1 - x²)
//   acos(x)   grad: -g / √(1 - x²)
//   atan(x)   grad: g / (1 + x²)

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Each trig op class: name maps to IBackend method via dispatch().
// sin/cos/tan → be.sin / be.cos / be.tan
// asin/acos/atan → be.asin / be.acos / be.atan

/// Autograd backward node for Sin.
class LUCID_API SinBackward : public UnaryOp<SinBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sin(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};
/// Sin.
LUCID_API TensorImplPtr sin_op(const TensorImplPtr& a);

/// Autograd backward node for Cos.
class LUCID_API CosBackward : public UnaryOp<CosBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cos(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};
/// Cos.
LUCID_API TensorImplPtr cos_op(const TensorImplPtr& a);

/// Autograd backward node for Tan.
class LUCID_API TanBackward : public UnaryOp<TanBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.tan(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};
/// Tan.
LUCID_API TensorImplPtr tan_op(const TensorImplPtr& a);

/// Autograd backward node for Asin.
class LUCID_API AsinBackward : public UnaryOp<AsinBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.asin(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};
/// Arcsin.
LUCID_API TensorImplPtr arcsin_op(const TensorImplPtr& a);

/// Autograd backward node for Acos.
class LUCID_API AcosBackward : public UnaryOp<AcosBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.acos(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};
/// Arccos.
LUCID_API TensorImplPtr arccos_op(const TensorImplPtr& a);

/// Autograd backward node for Atan.
class LUCID_API AtanBackward : public UnaryOp<AtanBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.atan(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};
/// Arctan.
LUCID_API TensorImplPtr arctan_op(const TensorImplPtr& a);

}  // namespace lucid
