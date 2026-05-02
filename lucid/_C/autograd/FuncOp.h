#pragma once

#include "AutogradNode.h"

namespace lucid {

template <class Derived, std::size_t N_IN>
using FuncOp = AutogradNode<Derived, N_IN>;

}
