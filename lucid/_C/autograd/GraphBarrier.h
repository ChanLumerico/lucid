// lucid/_C/autograd/GraphBarrier.h
//
// What the engine needs from a barrier node beyond Node's own interface: the
// gradients of a create_graph pass delivered slot by slot, and the identity of
// the backward pass that is running.  Both are found without touching Node, so
// its vtable — and with it the engine ABI — stays as it is.

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "../api.h"
#include "Node.h"

namespace lucid {

// The create_graph half of the barrier protocol.
//
// Eager backward hands a barrier each arriving gradient together with its
// slot (:meth:`Node::accumulate_barrier_grad`) and runs the barrier once,
// after every gradient has arrived.  Graph-mode backward kept one pending
// tensor per node and summed every arrival into it whatever its slot — right
// for a node with a single slot, wrong for one with several, whose gradients
// it added together.  A barrier that implements this interface receives its
// graph-mode gradients the way eager backward delivers them.
//
// The engine finds it with :func:`graph_barrier_of`; a barrier that does not
// implement it keeps the single-pending-tensor path.
class LUCID_API GraphBarrier {
public:
    virtual ~GraphBarrier() = default;

    // Add ``grad`` to what slot ``input_nr`` holds.  The sum is a graph op, so
    // it stays differentiable.
    virtual void accumulate_barrier_grad_for_graph(std::uint32_t input_nr, TensorImplPtr grad) = 0;

    // Run the barrier once every gradient has arrived: one tensor per outgoing
    // edge, null where no gradient flows.
    virtual std::vector<TensorImplPtr> apply_barrier_for_graph() = 0;
};

// The graph-mode barrier protocol of ``node``, or null.  An ordinary node
// costs one virtual call here, not a ``dynamic_cast``.
inline GraphBarrier* graph_barrier_of(Node* node) {
    return node != nullptr && node->is_barrier() ? dynamic_cast<GraphBarrier*>(node) : nullptr;
}

// The backward pass running on this thread.
//
// A barrier keeps the gradients delivered to it until its own turn comes.  A
// pass can deliver to a barrier and never give it that turn — ``autograd.grad``
// skips a node that leads to no requested input — and the next pass over the
// same graph (``retain_graph=True``) must not add onto what that one left.
// :meth:`Engine::backward` and :meth:`Engine::grad` each open a pass; a barrier
// holding gradients from a different pass discards them first.
//
// A pass opened inside another — a hook that calls ``autograd.grad`` — gets
// its own id and hands the outer one back when it closes, so the barriers of
// the outer pass keep what they hold.
class BackwardPass {
public:
    BackwardPass() noexcept : outer_(current_) {
        current_ = next_id_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    ~BackwardPass() { current_ = outer_; }

    BackwardPass(const BackwardPass&) = delete;
    BackwardPass& operator=(const BackwardPass&) = delete;

    // Id of the innermost pass open on this thread; ``0`` outside any pass.
    static std::uint64_t current() noexcept { return current_; }

private:
    std::uint64_t outer_;
    static inline thread_local std::uint64_t current_ = 0;
    static inline std::atomic<std::uint64_t> next_id_{0};
};

}  // namespace lucid
