#pragma once

#include "../autograd/AutogradNode.h"

namespace lucid {
namespace kernel {

template <class Derived, std::size_t N_IN>
using AutogradNode = ::lucid::AutogradNode<Derived, N_IN>;

}
}  // namespace lucid
