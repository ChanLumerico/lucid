// lucid/_C/autograd/GraphBarrier.h
//
// What the engine needs from a barrier node beyond Node's own interface: the
// gradients of a create_graph pass delivered slot by slot, and the identity of
// the backward pass that is running.  Both are found without touching Node, so
// its vtable — and with it the engine ABI — stays as it is.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <utility>
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
// its own id and hands the outer one back when it closes.  The outer pass is
// still open meanwhile, and what it delivered to a barrier the inner pass
// crosses is set aside rather than discarded (:class:`PassSlots`).
class BackwardPass {
public:
    BackwardPass() noexcept
        : id_(next_id_.fetch_add(1, std::memory_order_relaxed) + 1), outer_(innermost_) {
        innermost_ = this;
    }
    ~BackwardPass() { innermost_ = outer_; }

    BackwardPass(const BackwardPass&) = delete;
    BackwardPass& operator=(const BackwardPass&) = delete;

    // Id of the innermost pass open on this thread; ``0`` outside any pass.
    static std::uint64_t current() noexcept { return innermost_ != nullptr ? innermost_->id_ : 0; }

    // Whether pass ``id`` is open on this thread — running, or waiting for
    // a pass opened inside it to finish.
    static bool is_open(std::uint64_t id) noexcept {
        for (const BackwardPass* p = innermost_; p != nullptr; p = p->outer_) {
            if (p->id_ == id)
                return true;
        }
        return false;
    }

private:
    std::uint64_t id_;
    const BackwardPass* outer_;
    // The passes open on this thread, innermost first, linked through
    // ``outer_`` — a pointer, so the thread-local needs no initialisation.
    static inline thread_local const BackwardPass* innermost_ = nullptr;
    static inline std::atomic<std::uint64_t> next_id_{0};
};

// A barrier's slots, kept per backward pass.
//
// What a barrier is handed belongs to the pass that handed it.  A pass that
// delivered and never gave the barrier its turn — an exception, a pruned
// ``autograd.grad`` — must not seed the next pass.  A pass opened inside
// another that crosses the same barrier must neither add onto the outer
// pass's gradients nor wipe them: they are set aside while it runs and put
// back when the outer pass resumes, as each pass of the reference has input
// buffers of its own.
//
// The barrier keeps the running pass's slots where it reads them; this keeps
// the pass they belong to and those of outer passes still open.
template <typename Slots>
class PassSlots {
public:
    // Make ``live`` the running pass's slots.  When they belong to another
    // pass they are set aside if that pass is still open and dropped if it
    // is not; ``live`` becomes what the running pass set aside earlier, or
    // ``empty``.  Returns whether ``live`` changed hands.
    bool enter(Slots& live, Slots empty) {
        const std::uint64_t running = BackwardPass::current();
        if (running == pass_)
            return false;
        if (pass_ != 0 && BackwardPass::is_open(pass_))
            outer_.emplace_back(pass_, std::move(live));
        live = std::move(empty);
        for (auto it = outer_.begin(); it != outer_.end(); ++it) {
            if (it->first == running) {
                live = std::move(it->second);
                outer_.erase(it);
                break;
            }
        }
        std::erase_if(outer_, [](const auto& held) { return !BackwardPass::is_open(held.first); });
        pass_ = running;
        return true;
    }

    // Whether the slots the barrier holds are the running pass's already.
    bool is_running() const noexcept { return BackwardPass::current() == pass_; }

private:
    std::uint64_t pass_ = 0;
    std::vector<std::pair<std::uint64_t, Slots>> outer_;
};

}  // namespace lucid
