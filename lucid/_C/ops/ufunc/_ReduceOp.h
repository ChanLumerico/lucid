#pragma once

#include "../../kernel/ReduceKernel.h"

namespace lucid {

template <class Derived>
using ReduceOp = ReduceKernel<Derived>;

}
