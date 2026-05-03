// lucid/_C/ops/ufunc/Discrete.h
//
// Backward nodes for discontinuous (piecewise-constant) unary operations:
// round, floor, ceil, invert.  These functions have zero derivative almost
// everywhere (the derivative is undefined at integer boundaries) so all four
// set kHasGradient = false.  UnaryKernel::forward will skip autograd wiring
// entirely; grad_formula is provided only for completeness and returns an
// empty CpuStorage as a zero-gradient sentinel.  This matches PyTorch's
// behaviour for the same ops.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Backward node for element-wise round-to-nearest: y = round(x).
//
// No gradient; kHasGradient = false prevents autograd wiring.
// kSavesInput = false since there is nothing to save.
class LUCID_API RoundBackward : public UnaryOp<RoundBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.round(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr round_op(const TensorImplPtr& a);

// Backward node for element-wise floor: y = floor(x).
//
// No gradient; zero-gradient sentinel returned by grad_formula.
class LUCID_API FloorBackward : public UnaryOp<FloorBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.floor(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr floor_op(const TensorImplPtr& a);

// Backward node for element-wise ceiling: y = ceil(x).
//
// No gradient; zero-gradient sentinel returned by grad_formula.
class LUCID_API CeilBackward : public UnaryOp<CeilBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.ceil(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr ceil_op(const TensorImplPtr& a);

// Backward node for element-wise bitwise NOT: y = ~x.
//
// A bitwise operation is only valid on integer (non-floating-point) types,
// and has no meaningful gradient.  kHasGradient = false; AmpPolicy::KeepInput
// prevents float promotion so that the integer type is preserved.
class LUCID_API InvertBackward : public UnaryOp<InvertBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.invert(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

LUCID_API TensorImplPtr invert_op(const TensorImplPtr& a);

}  // namespace lucid
