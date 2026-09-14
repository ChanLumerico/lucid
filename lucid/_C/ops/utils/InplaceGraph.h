// lucid/_C/ops/utils/InplaceGraph.h
//
// What an in-place op has to do to the autograd graph, in one place.
//
// There are two in-place families — the unary one in ``ufunc/Inplace.cpp``
// and the binary one in ``bfunc/Inplace.cpp`` — and they had drifted:
// the unary side learned to adopt the forward's grad_fn and the binary
// side never did, so ``x.mul_(y)`` returned the gradient of whatever
// produced ``x`` and ``y`` received none at all.  Sharing the rules is
// what keeps the two from disagreeing again.

#pragma once

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "View.h"

namespace lucid::inplace {

// Refuse to mutate a leaf that requires grad.
//
// A leaf is where gradient accumulates, and an in-place write moves it
// without leaving anything behind that says so.  Nor is there a right
// number to return: after the write ``x`` *is* the result, so "the
// gradient with respect to x" names two different tensors depending on
// when it is asked.  The reference refuses the same setup for the same
// reason.  Callers that mean to overwrite say so with ``no_grad``, which
// is what every optimiser step already does.
inline void refuse_on_leaf(const TensorImplPtr& a, const char* name) {
    if (GradMode::is_enabled() && a->requires_grad() && !a->grad_fn())
        ErrorBuilder(name).fail("a leaf tensor that requires grad cannot be modified in place — "
                                "wrap the call in no_grad, or use the out-of-place form");
}

// A stand-in for ``a`` to run the forward against.
//
// A node that saves its input keeps a handle on the tensor it was given
// and graph-mode backward re-reads it — so once ``a``'s storage slot has
// been overwritten, ``sin_`` differentiated through
// ``grad(create_graph=True)`` computed ``cos(sin(x))`` instead of
// ``cos(x)``.  Eager ``backward()`` did not notice, because it saves a
// Storage by value at forward time, which is why the two routes
// disagreed and only in graph mode.
//
// The snapshot shares the buffer rather than copying it, and the caller's
// assignment replaces ``a``'s *slot* rather than the buffer, so the
// original values stay alive and unmutated for as long as the node needs
// them — and nothing is allocated when no graph is being built.  It also
// inherits where ``a`` sat, or the new node's parent is a fresh leaf and
// the chain back to the input is cut.
//
// A tensor that shares its buffer with a live view is the exception: its
// write goes into the buffer itself (``TensorImpl::write_through``), where
// a node holding a snapshot of that buffer would read the new values back
// — and the holder would stop the write besides.  While autograd records,
// its snapshot is a copy, whether or not ``a`` requires grad: the other
// operand of a binary op may, and its node saves this one.
inline TensorImplPtr snapshot(const TensorImplPtr& a) {
    const bool copy = a->is_aliased() && GradMode::is_enabled();
    if (!a->requires_grad() && !copy)
        return a;
    Storage storage = copy ? backend::Dispatcher::for_device(a->device())
                                 .clone(a->storage(), a->shape(), a->dtype())
                           : a->storage();
    auto source = std::make_shared<TensorImpl>(std::move(storage), a->shape(), a->dtype(),
                                               a->device(), a->requires_grad());
    if (a->requires_grad()) {
        source->set_grad_fn(a->grad_fn());
        source->set_grad_output_nr(a->grad_output_nr());
    }
    return source;
}

// Re-derive the other members of ``a``'s view family from ``a``.
//
// Every member reads the same bytes as ``a`` in the same order — each is a
// full-buffer reshape of the others — so after a write through ``a`` each
// one's values are ``a``'s, reshaped.  Autograd has to say so: a member
// that kept its old grad_fn would send its gradient to the values the
// write replaced.  When ``a`` requires grad each member becomes a reshape
// of it (a fresh ViewBackward); when it does not, a member that had a
// graph position is cut from it, since its values are a constant's now.
//
// A leaf that requires grad never gets here: ``write_through`` refuses a
// write to its family while autograd records.
inline void rebase_views(const TensorImplPtr& a) {
    // A write through a detached alias (``.data``) is untracked by design,
    // and a detached member takes no part in autograd either way.
    if (a->is_detached_alias())
        return;
    const bool differentiable = a->requires_grad();
    for (const auto& m : a->live_views()) {
        if (m->is_detached_alias())
            continue;
        // A leaf keeps its place: it is where gradient accumulates, and its
        // AccumulateGrad is a grad_fn this must not cut.
        if (m->is_leaf() && m->requires_grad())
            continue;
        if (differentiable) {
            kernel::NaryKernel<ViewBackward, 1>::wire_autograd({a}, m, false);
        } else if (m->grad_fn()) {
            m->set_grad_fn(nullptr);
            m->set_grad_output_nr(0);
            m->set_requires_grad(false);
        }
    }
}

// Move ``out``'s place in the autograd graph onto ``a``.
//
// ``fwd_fn`` builds a differentiable ``out`` whose grad_fn knows how to
// undo this op.  Taking only its storage kept the new numbers and threw
// the derivative away, so ``a`` still sat where it was before the call
// and reported the gradient of whatever produced it:
//
//     y = x * 1.0; y.exp_(); y.sum().backward()  ->  dx = 1
//                                     reference  ->  dx = exp(x)
//
// Silent, in both families: the value was right and only the derivative
// was not, so a model using one trained on a wrong gradient with nothing
// to show for it.  The views of ``a`` read the values it now holds, so
// they move with it.
inline bool adopt_graph_position(const TensorImplPtr& a, const TensorImplPtr& out) {
    if (!out->requires_grad() && !out->grad_fn())
        return false;
    a->set_requires_grad(true);
    a->set_grad_fn(out->grad_fn());
    a->set_grad_output_nr(out->grad_output_nr());
    if (a->is_aliased())
        rebase_views(a);
    return true;
}

// What to do when there was no graph position to adopt.
//
// Nothing to adopt means the op is not differentiable — ``ceil``,
// ``floor``, ``round`` and ``sign`` all end the graph.  ``a``'s contents
// no longer depend on what they were, so leaving its old grad_fn in place
// answers with the gradient of whatever produced ``a``, unchanged.  Cut
// it, which is the out-of-place convention — and cut its views with it,
// since they read the same constant values.
//
// Only under an active GradMode: inside ``no_grad`` the write is an
// ordinary mutation of a tensor belonging to a graph built earlier, and
// severing it there would lose a chain the caller means to keep — the
// version counter guards that case, and it still runs.
inline void detach_and_bump(const TensorImplPtr& a) {
    if (GradMode::is_enabled()) {
        if (a->grad_fn()) {
            a->set_grad_fn(nullptr);
            a->set_grad_output_nr(0);
            a->set_requires_grad(false);
        }
        if (a->is_aliased())
            rebase_views(a);
    }
    a->bump_version();
}

}  // namespace lucid::inplace
