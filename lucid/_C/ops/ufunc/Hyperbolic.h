// lucid/_C/ops/ufunc/Hyperbolic.h
//
// Autograd backward nodes and entry points for the hyperbolic family:
// sinh, cosh, tanh.  On CPU the backend routes to vForce (vvsinhf, vvcoshf,
// vvtanhf).  All ops use AmpPolicy::Promote.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Backward node for element-wise hyperbolic sine: y = sinh(x).
//
// Gradient rule: dL/dx = cosh(x) * dL/dy.
// Saves the input to evaluate cosh(x) in grad_formula.
class LUCID_API SinhBackward : public UnaryOp<SinhBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sinh(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise hyperbolic cosine: y = cosh(x).
//
// Gradient rule: dL/dx = sinh(x) * dL/dy.
// Saves the input to evaluate sinh(x) in grad_formula.
class LUCID_API CoshBackward : public UnaryOp<CoshBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cosh(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise hyperbolic tangent: y = tanh(x).
//
// Gradient rule: dL/dx = (1 - y^2) * dL/dy.
// Saves the *output* y instead of the input; squaring y is cheaper and avoids
// re-running vvtanhf.  kSavesInput = false explicitly opts out of the default
// input-save behaviour in UnaryKernel.
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

LUCID_API TensorImplPtr sinh_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr cosh_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr tanh_op(const TensorImplPtr& a);

}  // namespace lucid
