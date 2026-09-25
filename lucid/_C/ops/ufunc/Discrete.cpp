// lucid/_C/ops/ufunc/Discrete.cpp
//
// round, floor and ceil are step functions: their gradient is zero wherever
// it exists, and they stay in the graph with that zero gradient, as the
// reference framework keeps them.  invert (bitwise not) takes integers and
// bools only, so it never carries a gradient.

#include "Discrete.h"

#include "../../autograd/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../gfunc/Gfunc.h"

namespace lucid {

// round — KeepInput preserves integer types (round is a no-op on integers).
const OpSchema RoundBackward::schema_v1{"round", 1, AmpPolicy::KeepInput, true};

// Zero gradient: round is piecewise constant.
// A step function is flat wherever it is differentiable, so the gradient is
// zero — kept in the graph, as the reference framework keeps it.
Storage RoundBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr RoundBackward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr&,
                                               const TensorImplPtr&) {
    return zeros_like_op(g);
}
TensorImplPtr round_op(const TensorImplPtr& a) {
    return RoundBackward::forward(a);
}
LUCID_REGISTER_OP(RoundBackward)

// floor — KeepInput.
const OpSchema FloorBackward::schema_v1{"floor", 1, AmpPolicy::KeepInput, true};

// Zero gradient: floor is piecewise constant.
// A step function is flat wherever it is differentiable, so the gradient is
// zero — kept in the graph, as the reference framework keeps it.
Storage FloorBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr FloorBackward::grad_formula_impl(const TensorImplPtr& g,
                                               const TensorImplPtr&,
                                               const TensorImplPtr&) {
    return zeros_like_op(g);
}
TensorImplPtr floor_op(const TensorImplPtr& a) {
    return FloorBackward::forward(a);
}
LUCID_REGISTER_OP(FloorBackward)

// ceil — KeepInput.
const OpSchema CeilBackward::schema_v1{"ceil", 1, AmpPolicy::KeepInput, true};

// Zero gradient: ceil is piecewise constant.
// A step function is flat wherever it is differentiable, so the gradient is
// zero — kept in the graph, as the reference framework keeps it.
Storage CeilBackward::grad_formula(const Storage&) {
    return make_zero_storage(out_shape_, dtype_, device_);
}

TensorImplPtr CeilBackward::grad_formula_impl(const TensorImplPtr& g,
                                              const TensorImplPtr&,
                                              const TensorImplPtr&) {
    return zeros_like_op(g);
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
