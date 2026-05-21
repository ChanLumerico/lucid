// lucid/_C/autograd/Engine.h
//
// Declares Engine, the entry point for reverse-mode automatic differentiation.
// Callers invoke Engine::backward() with the scalar root tensor to start a
// backward pass; the engine handles graph traversal, gradient propagation, and
// cleanup internally.

#pragma once

#include <memory>

#include "../api.h"
#include "../core/TensorImpl.h"

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
};

}  // namespace lucid
