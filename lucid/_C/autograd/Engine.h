// lucid/_C/autograd/Engine.h
//
// Declares Engine, the entry point for reverse-mode automatic differentiation.
// Callers invoke Engine::backward() with the scalar root tensor to start a
// backward pass; the engine handles graph traversal, gradient propagation, and
// cleanup internally.

#pragma once

#include <memory>
#include <vector>

#include "../api.h"
#include "../core/TensorImpl.h"
#include "../core/fwd.h"

namespace lucid {

// Reverse-mode autograd engine — drives the backward pass over a recorded
// computation graph rooted at a single output :class:`TensorImpl`.
//
// ``Engine`` is a stateless class that exposes a single static entry point
// (:func:`Engine::backward`).  It is never instantiated; all per-call state
// (worklist, accumulated gradients, visited set) lives on the stack of the
// invoking thread.
//
// Algorithm
// ---------
// 1. Compute a reverse-topological ordering of the backward graph using an
//    iterative post-order DFS (the recursive form would overflow on deep
//    networks).
// 2. Optionally run a fusion pass that collapses adjacent backward nodes
//    (e.g. ``LinearBackward`` + ``ReluBackward``) into a single fused node.
// 3. Walk the ordering once, calling each node's ``apply()``; gradients
//    arriving at a node from multiple producers are summed into a pending
//    map before the node is executed.
// 4. Leaves (tensors with no ``grad_fn``) are reached through their
//    :class:`AccumulateGrad` sentinel, which writes the final gradient into
//    ``leaf.grad``.
//
// Thread Safety
// -------------
// Two concurrent ``backward`` calls on the same graph are undefined
// behaviour — nodes are consumed and (when ``retain_graph=false``) destroyed
// exactly once.  Independent graphs may be driven from separate threads.
//
// See Also
// --------
// :class:`AccumulateGrad` : terminal node that writes into ``leaf.grad``.
// :class:`Node` : abstract base for every backward graph node.
class LUCID_API Engine {
public:
    // Run reverse-mode automatic differentiation starting from ``root``.
    //
    // Walks the computation graph attached to ``root->grad_fn()`` in reverse
    // topological order, computes per-edge input gradients via each node's
    // ``apply()`` method, accumulates contributions at branch points, and
    // hands terminal gradients to :class:`AccumulateGrad` for in-place
    // accumulation into ``leaf.grad``.
    //
    // Parameters
    // ----------
    // root : std::shared_ptr<TensorImpl>
    //     Output tensor to differentiate.  Must be non-null.  If ``root``
    //     has no ``grad_fn`` it is treated as a leaf and the seed is
    //     accumulated directly into ``root->grad``.
    // grad_seed : Storage, optional
    //     Initial gradient injected at ``root``.  An empty ``Storage``
    //     (default) is replaced by a ones-tensor of ``root``'s shape /
    //     dtype / device — the common case of differentiating a scalar
    //     loss.
    // retain_graph : bool, optional
    //     When ``false`` (default), each node's ``release_saved()`` is
    //     called immediately after its ``apply()`` and ``root->grad_fn``
    //     is cleared on return, so a second ``backward()`` call is
    //     impossible.  Pass ``true`` to preserve the graph for multiple
    //     backward calls.
    // create_graph : bool, optional
    //     When ``true``, the backward pass itself is recorded in the
    //     autograd graph so that higher-order gradients can be taken on
    //     the resulting ``.grad`` tensors.  Implies ``retain_graph=true``
    //     because the forward nodes are re-used by the new graph.
    //     Concrete nodes must override ``apply_for_graph`` for this mode;
    //     nodes lacking graph support raise a clear error naming the op.
    //
    // Raises
    // ------
    // std::runtime_error
    //     If ``root`` is null, if a node returns an ``input_grads``
    //     vector whose size disagrees with its outgoing edges (and both
    //     are non-empty), or if ``validate_versions()`` detects an
    //     in-place mutation of a saved input tensor.
    //
    // Notes
    // -----
    // The engine consumes nodes destructively when ``retain_graph=false``:
    // ``release_saved()`` frees the forward tensors each node had stashed
    // for its backward formula, and ``clear_grad_fn()`` on ``root``
    // severs the producer→graph reference so the chain of shared_ptrs
    // collapses.
    //
    // Examples
    // --------
    // >>> # Pseudo-C++: differentiate a scalar loss
    // >>> Engine::backward(loss);
    // >>> // Now every leaf with requires_grad=true has its .grad populated.
    static void backward(const std::shared_ptr<TensorImpl>& root,
                         Storage grad_seed = Storage{CpuStorage{}},
                         bool retain_graph = false,
                         bool create_graph = false);

    // Differentiate ``root`` with respect to ``inputs`` without writing any
    // ``.grad``.
    //
    // The functional counterpart to :func:`backward`.  It walks the same
    // graph, but instead of letting gradients terminate in the
    // :class:`AccumulateGrad` nodes that own each leaf's ``.grad`` slot, it
    // intercepts them and returns them to the caller.  No tensor's gradient
    // state is read or modified — not the requested inputs', and not any
    // other leaf's.
    //
    // That last part is the reason this exists.  Emulating it in Python by
    // running a full :func:`backward` and then restoring ``.grad`` can only
    // restore the tensors the caller named; every *other* leaf in the graph
    // keeps whatever the traversal deposited, which silently corrupts
    // parameters the caller never mentioned.
    //
    // Parameters
    // ----------
    // root : const std::shared_ptr<TensorImpl>&
    //     Output to differentiate.  Must be non-null.
    // grad_seed : Storage
    //     Upstream gradient for ``root``.  An empty storage means "ones",
    //     matching :func:`backward`.
    // inputs : const std::vector<std::shared_ptr<TensorImpl>>&
    //     Tensors to differentiate with respect to.  May be leaves or
    //     interior nodes; entries must be non-null.
    // retain_graph : bool, default=false
    //     Keep saved forward tensors so the graph can be traversed again.
    // create_graph : bool, default=true for the returned gradients to be
    //     differentiable in turn (higher-order).  Implies ``retain_graph``.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     One entry per requested input, in order.  An entry is null when
    //     that input lies outside ``root``'s graph — the caller decides
    //     whether that is an error.
    //
    // Raises
    // ------
    // LucidError
    //     If ``root`` or any entry of ``inputs`` is null.
    //
    // Notes
    // -----
    // A leaf is matched by the :class:`AccumulateGrad` node bound to it; an
    // interior tensor is matched by its own ``grad_fn``.  Interior captures
    // do not stop the traversal, so asking for a tensor halfway down the
    // graph still lets gradients reach everything below it.
    //
    // See Also
    // --------
    // :func:`Engine::backward` — the accumulating counterpart.
    static std::vector<TensorImplPtr> grad(const std::shared_ptr<TensorImpl>& root,
                                           Storage grad_seed,
                                           const std::vector<std::shared_ptr<TensorImpl>>& inputs,
                                           bool retain_graph = false,
                                           bool create_graph = false);
};

}  // namespace lucid
