// lucid/_C/ops/ufunc/Exponential.h
//
// Autograd backward nodes and entry points for the exponential and logarithmic
// family: exp, log, log2, sqrt.  On CPU, the backend routes to vForce
// (vvexpf, vvlogf, …) for SIMD throughput.  All ops request AmpPolicy::
// ForceFP32 (exp/log/log2) or AmpPolicy::Promote (sqrt) so that
// lower-precision inputs are upcast before computation.

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

// Backward node for element-wise exp: y = e^x.
//
// Gradient rule: dL/dx = y * dL/dy.
// Saves the *output* rather than the input because the backward formula uses y,
// not x, avoiding a redundant re-evaluation of exp.
class LUCID_API ExpBackward : public UnaryOp<ExpBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.exp(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise natural logarithm: y = ln(x).
//
// Gradient rule: dL/dx = dL/dy / x.
// Saves the input so that grad_formula can divide by x.
class LUCID_API LogBackward : public UnaryOp<LogBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.log(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise base-2 logarithm: y = log2(x).
//
// Gradient rule: dL/dx = dL/dy / (x * ln(2)).
// The ln(2) constant is embedded in grad_formula as a high-precision literal.
class LUCID_API Log2Backward : public UnaryOp<Log2Backward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.log2(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Backward node for element-wise square root: y = sqrt(x).
//
// Gradient rule: dL/dx = dL/dy / (2 * y).
// Saves the *output* y because that avoids recomputing sqrt(x) and is
// numerically identical.
class LUCID_API SqrtBackward : public UnaryOp<SqrtBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.sqrt(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr exp_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr log_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr log2_op(const TensorImplPtr& a);

LUCID_API TensorImplPtr sqrt_op(const TensorImplPtr& a);

}  // namespace lucid
