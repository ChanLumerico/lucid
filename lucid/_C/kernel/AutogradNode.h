#pragma once

// =====================================================================
// Lucid C++ engine — kernel/AutogradNode.h
// =====================================================================
//
// Re-export of autograd/AutogradNode.h for kernel/ layer consumers.
// The actual implementation lives in autograd/ (rank 3).
// kernel/ (rank 4) may include from autograd/ without layer violation.
//
// Layer: kernel/. Includes from autograd/ (lower rank — OK).

#include "../autograd/AutogradNode.h"

// Re-export into kernel:: namespace for code that uses
// `kernel::AutogradNode<D, N>` explicitly.
namespace lucid {
namespace kernel {

template <class Derived, std::size_t N_IN>
using AutogradNode = ::lucid::AutogradNode<Derived, N_IN>;

}  // namespace kernel
}  // namespace lucid
