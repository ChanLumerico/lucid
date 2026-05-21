// lucid/_C/autograd/FuncOp.h
//
// Type alias :class:`FuncOp` — the canonical base every concrete backward
// node in the ``ops/`` and ``nn/`` layers inherits from.  Currently aliases
// :class:`AutogradNode` directly; the indirection exists so that a future
// split between "functional op backward" and the lower-level
// :class:`AutogradNode` boilerplate can be introduced without touching every
// op file.

#pragma once

#include "AutogradNode.h"

namespace lucid {

// CRTP alias for :class:`AutogradNode` used by every concrete backward
// node in the engine.
//
// ``FuncOp<Derived, N_IN>`` is identical to ``AutogradNode<Derived, N_IN>``
// — see :class:`AutogradNode` for the full attribute list and lifecycle
// contract.  Concrete nodes inherit through this alias so that their
// declarations read as "a functional op backward":
//
//     class LinearBackward : public FuncOp<LinearBackward, 3> { ... };
//     class Conv2dBackward : public FuncOp<Conv2dBackward, 3> { ... };
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete backward class (CRTP self-type).  Must expose a
//     ``static constexpr OpSchema schema_v1`` and a ``public static``
//     ``forward(...)`` returning :class:`TensorImplPtr` (or a tuple
//     thereof) plus an :meth:`apply` override returning
//     ``std::vector<Storage>`` of length ``N_IN``.
// N_IN : std::size_t
//     Number of forward input tensors whose :class:`Storage` is preserved
//     in :attr:`AutogradNode::saved_inputs_` for use during backward.
//
// Notes
// -----
// **Static schema.** ``Derived::schema_v1`` is the registered
// :class:`OpSchema` exposing the op's name, version tag, and AMP policy
// (autocast cast targets) to the dispatcher.  The ``_v1`` suffix is the
// schema-version namespace — when an op's wire format changes, a new
// ``schema_v2`` is added alongside the old one to preserve checkpoint
// compatibility.
//
// **Forward / apply pairing.** The static ``Derived::forward(...)`` runs
// the op via the appropriate backend kernel, populates
// :attr:`saved_inputs_` / :attr:`saved_output_` / :attr:`input_tensors_`
// in slot order, calls :meth:`Node::set_next_edges` and
// :meth:`Node::set_saved_versions`, and attaches the constructed node as
// ``grad_fn`` on the output :class:`TensorImpl`.  During backward,
// :meth:`apply(grad_out)` consumes those saved values once and returns one
// :class:`Storage` per forward input — empty for inputs that do not
// require grad.
//
// **AMP interaction.** ``Derived::schema_v1`` carries an
// :class:`AmpPolicy` enum tag — :data:`AmpPolicy::PromoteToF32`,
// :data:`AmpPolicy::PreserveF16`, etc.  The autocast layer reads the
// policy when deciding whether to upcast inputs before calling
// :meth:`forward`; backward inherits the cast targets via the saved
// :class:`Storage` dtypes, so :meth:`apply` does not need additional AMP
// logic.
//
// **Slot ordering is wire format.** ``saved_[i]`` in the derived class
// must correspond to forward input ``i`` for the lifetime of the schema
// version — checkpoint files persist gradients in that exact order.
// Reordering inputs requires a new ``schema_v2``.
//
// See Also
// --------
// :class:`AutogradNode` — the underlying CRTP base, with full attribute
//     and lifecycle documentation.
// :class:`Node` — the abstract root with the virtual :meth:`apply` /
//     :meth:`release_saved` contract.
template <class Derived, std::size_t N_IN>
using FuncOp = AutogradNode<Derived, N_IN>;

}  // namespace lucid
