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

// Terminal backward node that accumulates an incoming gradient into a leaf tensor.
//
// An AccumulateGrad node is attached to a leaf TensorImpl's grad_fn slot the
// first time that tensor is used in a computation requiring gradients (see
// _register_python_backward_node and the op builders in ops/).  Leaf tensors
// do not themselves produce a grad_fn during their own "forward" step, so
// AccumulateGrad acts as the sentinel that terminates the backward walk.
//
// On each call to apply():
//   - If the leaf has been garbage-collected (weak_ptr expired), nothing
//     happens (the result is discarded).
//   - If the leaf does not require gradients, nothing happens.
//   - Otherwise, if no gradient has been accumulated yet the incoming grad is
//     moved in directly; subsequent calls add (+=) the incoming grad to the
//     existing grad using accumulate_into().
//
// AccumulateGrad holds only a weak_ptr to the leaf so that it does not
// artificially extend the tensor's lifetime while the backward graph is alive.
// next_edges() is always empty — there is nowhere further to propagate.
class AccumulateGrad : public Node {
public:
    // Construct an AccumulateGrad node that will write gradients into leaf.
    // leaf must be a leaf TensorImpl (is_leaf() == true); this is not
    // asserted here but is a precondition enforced at the call sites.
    explicit AccumulateGrad(std::weak_ptr<TensorImpl> leaf);

    // Add grad_out into the leaf tensor's .grad storage.
    // Always returns an empty vector because there are no further edges.
    std::vector<Storage> apply(Storage grad_out) override;

    // Accessor for the leaf weak_ptr, used primarily by tests and
    // graph-inspection utilities.
    std::weak_ptr<TensorImpl> leaf() const { return leaf_; }

private:
    std::weak_ptr<TensorImpl> leaf_;
};

}  // namespace lucid
