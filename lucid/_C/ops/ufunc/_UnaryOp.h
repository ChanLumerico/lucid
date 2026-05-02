#pragma once

#include "../../kernel/UnaryKernel.h"

namespace lucid {

template <class Derived>
using UnaryOp = UnaryKernel<Derived>;

}
