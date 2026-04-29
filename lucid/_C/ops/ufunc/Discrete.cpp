#include "Discrete.h"

#include "../../core/OpRegistry.h"

namespace lucid {

// --------------- Round (banker's, half-to-even) ---------------
const OpSchema RoundBackward::schema_v1{"round", 1, AmpPolicy::KeepInput, true};

Storage RoundBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr round_op(const TensorImplPtr& a) {
    return RoundBackward::forward(a);
}
LUCID_REGISTER_OP(RoundBackward)

// --------------- Floor ---------------
const OpSchema FloorBackward::schema_v1{"floor", 1, AmpPolicy::KeepInput, true};

Storage FloorBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr floor_op(const TensorImplPtr& a) {
    return FloorBackward::forward(a);
}
LUCID_REGISTER_OP(FloorBackward)

// --------------- Ceil ---------------
const OpSchema CeilBackward::schema_v1{"ceil", 1, AmpPolicy::KeepInput, true};

Storage CeilBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr ceil_op(const TensorImplPtr& a) {
    return CeilBackward::forward(a);
}
LUCID_REGISTER_OP(CeilBackward)

// --------------- Invert (bitwise NOT, integer-only) ---------------
const OpSchema InvertBackward::schema_v1{"invert", 1, AmpPolicy::KeepInput, true};

Storage InvertBackward::grad_formula(const Storage&) {
    return Storage{CpuStorage{}};
}
TensorImplPtr invert_op(const TensorImplPtr& a) {
    return InvertBackward::forward(a);
}
LUCID_REGISTER_OP(InvertBackward)

}  // namespace lucid
