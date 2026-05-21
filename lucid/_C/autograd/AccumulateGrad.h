// lucid/_C/autograd/AccumulateGrad.h
//
// Declares AccumulateGrad, the terminal Node placed at every leaf tensor that
// requires a gradient.  It is the last node visited by the Engine during any
// backward pass and is responsible for writing the accumulated gradient into
// the leaf's TensorImpl.

#pragma once

#include <memory>

#include "../core/TensorImpl.h"
#include "Node.h"

namespace lucid {

// Terminal backward node that accumulates an incoming gradient into a leaf
// tensor's ``.grad`` storage.
//
// An ``AccumulateGrad`` is attached to a leaf :class:`TensorImpl`'s
// ``grad_fn`` slot the first time the tensor is used in a computation that
// requires gradients (see ``_register_python_backward_node`` and the op
// builders under ``ops/``).  Leaves do not themselves produce a ``grad_fn``
// during their own "forward" step, so this sentinel terminates the backward
// walk: its ``next_edges()`` is always empty.
//
// Apply Semantics
// ---------------
// On each call to ``apply(grad_out)``:
//
// 1. If the leaf has been garbage-collected (``weak_ptr`` expired), the
//    gradient is silently discarded.
// 2. If the leaf no longer requires gradients (e.g. ``requires_grad_(False)``
//    was called between forward and backward), the gradient is discarded.
// 3. Incoming gradients are cast to the leaf's own dtype to handle AMP /
//    autocast paths where the same parameter may be reached via different
//    effective dtypes (e.g. F16 from a Conv and F32 from a ForceFP32 branch).
// 4. If the leaf has no gradient yet, ``grad_out`` is moved in directly —
//    avoiding the cost of allocating a zero buffer and adding to it.
//    Otherwise ``accumulate_into()`` performs an in-place ``+=`` using the
//    appropriate backend (Accelerate on CPU, MLX on GPU).
//
// Ownership
// ---------
// ``AccumulateGrad`` holds only a ``weak_ptr`` to its leaf so that it does
// not artificially extend the tensor's lifetime while the backward graph is
// alive.  Non-leaf intermediate tensors never carry an ``AccumulateGrad`` —
// their gradient flows through whatever produced them and is only stored
// transiently in the engine's ``pending`` map (or in ``.grad`` if the user
// explicitly called ``retain_grad`` on the intermediate).
//
// See Also
// --------
// :class:`Engine` : reverse-mode driver that dispatches into this node.
// :class:`Node` : abstract base class.
class AccumulateGrad : public Node {
public:
    // Construct an ``AccumulateGrad`` bound to a specific leaf tensor.
    //
    // Parameters
    // ----------
    // leaf : std::weak_ptr<TensorImpl>
    //     The leaf tensor whose ``.grad`` slot this node writes into.
    //     Must satisfy ``leaf->is_leaf() == true``; not asserted here but
    //     enforced at all call sites (op builders, hook-edge constructors).
    explicit AccumulateGrad(std::weak_ptr<TensorImpl> leaf);

    // Write ``grad_out`` into the leaf's gradient storage.
    //
    // The first call moves the gradient in directly; subsequent calls
    // accumulate ``+=`` in place via the backend-appropriate kernel.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient.  Will be cast to the leaf's dtype if it
    //     does not already match (handles AMP mixed-dtype paths).
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Always empty — ``AccumulateGrad`` has no outgoing edges and the
    //     engine has no further work to schedule for this branch.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode variant invoked when ``Engine::backward`` runs with
    // ``create_graph=true``.  Stores ``grad_out`` (a :class:`TensorImpl`
    // that may carry its own ``grad_fn``) into the leaf's ``grad_impl``
    // slot so the gradient tensor itself remains differentiable for
    // higher-order gradient computations.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Gradient tensor whose autograd metadata must be preserved.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     Always empty for the same reason as :func:`apply`.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

    // Human-readable node name surfaced in error messages and graph dumps.
    std::string node_name() const override { return "AccumulateGrad"; }

    // Accessor for the leaf ``weak_ptr``.
    //
    // Returns
    // -------
    // std::weak_ptr<TensorImpl>
    //     A weak reference to the bound leaf — may be expired.  Used by
    //     tests and graph-inspection utilities.
    std::weak_ptr<TensorImpl> leaf() const { return leaf_; }

private:
    std::weak_ptr<TensorImpl> leaf_;
};

}  // namespace lucid
