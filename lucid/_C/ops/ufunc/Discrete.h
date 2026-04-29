#pragma once

// =====================================================================
// Lucid C++ engine — discrete / non-differentiable unary ops.
// =====================================================================
//
// All four ops have zero gradient (or no gradient at all on integer inputs).
// kHasGradient = false on the CRTP base skips graph wiring entirely.
//
//   round(x)   — banker's rounding (half-to-even, matches np.round)
//   floor(x)
//   ceil(x)
//   invert(x)  — bitwise NOT (integer dtypes)

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

/// Autograd backward node for Round.
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
/// Round.
LUCID_API TensorImplPtr round_op(const TensorImplPtr& a);

/// Autograd backward node for Floor.
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
/// Floor.
LUCID_API TensorImplPtr floor_op(const TensorImplPtr& a);

/// Autograd backward node for Ceil.
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
/// Ceil.
LUCID_API TensorImplPtr ceil_op(const TensorImplPtr& a);

/// Autograd backward node for Invert.
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
/// Invert.
LUCID_API TensorImplPtr invert_op(const TensorImplPtr& a);

}  // namespace lucid
