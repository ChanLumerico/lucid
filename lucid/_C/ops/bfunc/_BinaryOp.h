// lucid/_C/ops/bfunc/_BinaryOp.h
//
// Thin alias that makes BinaryKernel<Derived> available under the shorter name
// BinaryOp<Derived> throughout the binary-operation subsystem.  Every concrete
// backward node in ops/bfunc/ (AddBackward, SubBackward, …) inherits from this
// alias so that the subsystem does not need to reach back into the kernel/
// directory directly.  The full implementation of the CRTP base — broadcasting,
// autograd wiring, forward dispatch, and the apply() trampoline — lives in
// kernel/BinaryKernel.h.

#pragma once

#include "../../kernel/BinaryKernel.h"

namespace lucid {

template <class Derived>
using BinaryOp = BinaryKernel<Derived>;

}
