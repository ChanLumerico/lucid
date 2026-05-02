#pragma once

// =====================================================================
// Lucid C++ engine — hyperbolic unary ops.
// =====================================================================
//
//   sinh(x)   grad: cosh(x) * g
//   cosh(x)   grad: sinh(x) * g
//   tanh(x)   grad: (1 - tanh²(x)) * g  =  (1 - z²) * g  [z = output]

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

/// Autograd backward node for Sinh.
class LUCID_API SinhBackward : public UnaryOp<SinhBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sinh(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

/// Autograd backward node for Cosh.
class LUCID_API CoshBackward : public UnaryOp<CoshBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cosh(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

/// Autograd backward node for Tanh.
class LUCID_API TanhBackward : public UnaryOp<TanhBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.tanh(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

/// Sinh.
LUCID_API TensorImplPtr sinh_op(const TensorImplPtr& a);
/// Cosh.
LUCID_API TensorImplPtr cosh_op(const TensorImplPtr& a);
/// Tanh.
LUCID_API TensorImplPtr tanh_op(const TensorImplPtr& a);

}  // namespace lucid
