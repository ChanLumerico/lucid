// lucid/_C/kernel/AutogradNode.h
//
// Re-export of the core AutogradNode template into the kernel namespace.
// All CRTP kernel bases (UnaryKernel, BinaryKernel, NaryKernel,
// ReduceKernel) inherit from kernel::AutogradNode<Derived, N_IN>, which
// resolves to the same lucid::AutogradNode<Derived, N_IN> defined in
// autograd/AutogradNode.h. This alias keeps the kernel headers free of
// direct autograd/ include paths in downstream ops/ code.

#pragma once

#include "../autograd/AutogradNode.h"

namespace lucid {
namespace kernel {

// Alias that maps kernel::AutogradNode<Derived, N_IN> to the canonical
// autograd node template. N_IN is the number of input tensors the op
// accepts, which determines the size of saved_inputs_ and the edge count
// wired into the autograd graph during forward().
template <class Derived, std::size_t N_IN>
using AutogradNode = ::lucid::AutogradNode<Derived, N_IN>;

}
}  // namespace lucid
