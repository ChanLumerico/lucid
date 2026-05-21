// lucid/_C/tensor/AutogradMeta.h
//
// Optional autograd bookkeeping embedded in :class:`TensorImpl` — present
// only when a tensor participates in differentiation.
//
// A :class:`TensorImpl` carries an :class:`AutogradMeta` only when it has
// been marked ``requires_grad = true`` or when it is the output of an op
// that produces a differentiable result.  Tensors that never participate
// in differentiation (integer index tensors, random samples, buffers)
// have no :class:`AutogradMeta` to avoid the overhead of allocating the
// struct and populating the optional grad storage.
//
// Notes
// -----
// The version counter is incremented by :func:`TensorImpl::bump_version`
// on every in-place modification (optimizer step, in-place ops).  Any
// autograd node that captured this tensor during ``forward()`` records
// the version at capture time; :func:`validate_versions` in the backward
// pass checks for mismatches to detect illegal in-place mutations and
// raises :class:`VersionMismatch`.
//
// See Also
// --------
// :class:`AutogradNode` — backward node referenced by :attr:`grad_fn`.
// :class:`AccumulateGrad` — terminal node that writes into :attr:`grad`.
// :class:`TensorImpl`     — owner that holds this struct in an
//     ``std::optional<AutogradMeta>`` slot.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class TensorImpl;

// Per-tensor autograd metadata — gradient buffer, backward edge, and
// version counter for in-place safety.
//
// ``requires_grad`` controls whether operations involving this tensor
// build an autograd graph.  ``is_leaf`` is ``true`` for parameters
// created by the user (model weights) and ``false`` for intermediate
// tensors produced by ops.  Leaf tensors accumulate gradients into
// :attr:`grad`; non-leaf tensors do not accumulate unless the engine
// later honours a ``retain_grad`` opt-in (tracked in the sibling
// :class:`core::AutogradMeta`, not yet exposed in this struct).
//
// :attr:`grad_fn` points to the backward Node that produced this
// tensor.  It is ``nullptr`` for leaf tensors (which use
// :class:`AccumulateGrad` installed lazily by ``ensure_grad_fn``) and
// for tensors detached from the graph via :func:`detach`.
//
// Attributes
// ----------
// requires_grad : bool
//     Whether autograd tracks this tensor.  When ``true``, downstream
//     ops emit an :class:`AutogradNode` and link it into :attr:`grad_fn`
//     of their outputs.
// is_leaf : bool
//     ``true`` for user-created parameters (no producing op); ``false``
//     for op outputs.  Leaves install :class:`AccumulateGrad` as their
//     ``grad_fn`` on first backward and write into :attr:`grad`.
// version : std::int64_t
//     Monotonically increasing in-place mutation counter.  Bumped by
//     :func:`TensorImpl::bump_version` on every in-place op or
//     optimizer step.  Saved tensors snapshot this value at capture
//     time and compare it during backward.
// grad_fn : NodePtr
//     Backward function (autograd graph edge) that produced this
//     tensor.  ``nullptr`` for leaves and detached tensors.
// grad : std::optional<Storage>
//     Accumulated gradient storage, populated on leaves after a normal
//     :func:`backward` call.  Lazily allocated by
//     :class:`AccumulateGrad` on first invocation; cleared by
//     :func:`zero_grad`.
// grad_impl : std::shared_ptr<TensorImpl>
//     Full :class:`TensorImpl` view of the gradient — populated
//     instead of (or in addition to) :attr:`grad` when
//     ``backward(create_graph=true)`` is used.  Allows the gradient
//     itself to participate in further autograd operations
//     (second-order derivatives, MAML, Hessian-vector products, …).
//
// Notes
// -----
// Invariants enforced by the autograd layer:
//
//   * ``is_leaf == true``  ⇒ ``grad_fn == nullptr``.
//   * ``is_leaf == false`` ⇒ ``grad_fn != nullptr``.
//   * :attr:`version` is monotonically non-decreasing for the lifetime
//     of the owning :class:`TensorImpl`.
//
// Lifecycle: created at first parameter use (when ``requires_grad`` is
// set), persists through ``forward()`` / ``backward()`` cycles, and
// :attr:`grad` is cleared by ``optimizer.zero_grad()`` between
// iterations.
//
// See Also
// --------
// :class:`AutogradNode` — backward node type referenced by
//     :attr:`grad_fn`.
// :class:`AccumulateGrad` — terminal node that writes into
//     :attr:`grad` for leaf tensors.
struct AutogradMeta {
    // Whether this tensor participates in gradient computation.
    bool requires_grad = false;
    // True for user-created parameters; false for op outputs.
    bool is_leaf = true;
    // Monotonically increasing in-place mutation counter.
    std::int64_t version = 0;
    // Backward function that produced this tensor (null for leaves).
    NodePtr grad_fn;
    // Accumulated gradient storage; set on leaves after normal backward().
    std::optional<Storage> grad;
    // Gradient as a full TensorImpl when backward was run with create_graph=true.
    // Allows the gradient itself to participate in further autograd operations
    // (second-order derivatives, MAML, Hessian-vector products, etc.).
    std::shared_ptr<TensorImpl> grad_impl;
};

}  // namespace lucid
