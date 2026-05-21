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

// Namespace-local alias of the core CRTP autograd node template.
//
// This alias is what every kernel CRTP base (``UnaryKernel``,
// ``BinaryKernel``, ``NaryKernel``, ``ReduceKernel``) inherits from. It
// resolves to the identical type as ``::lucid::AutogradNode<Derived, N_IN>``
// — the indirection exists purely so kernel headers can refer to
// ``kernel::AutogradNode`` and keep the autograd include path implicit
// rather than re-exposing ``autograd/AutogradNode.h`` to every downstream
// op translation unit.
//
// Template Parameters
// -------------------
// Derived : class
//     CRTP self-type of the concrete backward node (e.g.
//     ``SumBackward``, ``AddBackward``). Must expose a
//     ``static constexpr OpSchema schema_v1`` member.
// N_IN : std::size_t
//     Number of input tensors the forward op consumes. Determines the
//     compile-time array sizes for ``saved_inputs_``, ``input_shapes_``,
//     ``input_tensors_``, and the edge count wired into the autograd
//     graph during ``forward()``.
//
// Notes
// -----
// Because this is a using-alias (not a derived template), there is no
// behavioural difference between ``kernel::AutogradNode<D, N>`` and
// ``::lucid::AutogradNode<D, N>`` — they share saved-input lifecycle,
// version validation, and the ``release_saved`` implementation
// documented on the canonical type.
//
// See Also
// --------
// :class:`::lucid::AutogradNode` — the canonical implementation.
// :class:`UnaryKernel`, :class:`BinaryKernel`, :class:`NaryKernel`,
// :class:`ReduceKernel` — CRTP kernel bases that consume this alias.
template <class Derived, std::size_t N_IN>
using AutogradNode = ::lucid::AutogradNode<Derived, N_IN>;

}  // namespace kernel
}  // namespace lucid
