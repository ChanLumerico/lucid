// lucid/_C/ops/ufunc/Discrete.cpp
//
// All discrete ops share the same no-op grad_formula: return an empty
// CpuStorage{}.  Because kHasGradient = false, UnaryKernel::forward never
// calls apply() or grad_formula in practice; the body is present only to
// satisfy the interface contract.

#include "Discrete.h"

#include "../../core/OpRegistry.h"

namespace lucid {

// round — KeepInput preserves integer types (round is a no-op on integers).
const OpSchema RoundBackward::schema_v1{"round", 1, AmpPolicy::KeepInput, true};

// Zero gradient: round is piecewise constant.
Storage RoundBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr round_op(const TensorImplPtr& a) {
    return RoundBackward::forward(a);
}
LUCID_REGISTER_OP(RoundBackward)

// floor — KeepInput.
const OpSchema FloorBackward::schema_v1{"floor", 1, AmpPolicy::KeepInput, true};

// Zero gradient: floor is piecewise constant.
Storage FloorBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr floor_op(const TensorImplPtr& a) {
    return FloorBackward::forward(a);
}
LUCID_REGISTER_OP(FloorBackward)

// ceil — KeepInput.
const OpSchema CeilBackward::schema_v1{"ceil", 1, AmpPolicy::KeepInput, true};

// Zero gradient: ceil is piecewise constant.
Storage CeilBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr ceil_op(const TensorImplPtr& a) {
    return CeilBackward::forward(a);
}
LUCID_REGISTER_OP(CeilBackward)

// invert — KeepInput; bitwise NOT is only defined for integer types.
const OpSchema InvertBackward::schema_v1{"invert", 1, AmpPolicy::KeepInput, true};

// Zero gradient: bitwise NOT has no floating-point derivative.
Storage InvertBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr invert_op(const TensorImplPtr& a) {
    return InvertBackward::forward(a);
}
LUCID_REGISTER_OP(InvertBackward)

}  // namespace lucid
