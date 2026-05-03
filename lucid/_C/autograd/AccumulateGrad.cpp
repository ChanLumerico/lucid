// lucid/_C/autograd/AccumulateGrad.cpp
//
// Implements AccumulateGrad::apply(), which is the final gradient sink for
// every leaf tensor that participates in a backward pass.

#include "AccumulateGrad.h"

#include <utility>

#include "Helpers.h"

namespace lucid {

AccumulateGrad::AccumulateGrad(std::weak_ptr<TensorImpl> leaf) : leaf_(std::move(leaf)) {}

// Write grad_out into the leaf's gradient storage.
//
// Three early-exit cases:
//   1. The leaf TensorImpl has been destroyed (weak_ptr expired) — nothing
//      to accumulate into; silently discard the gradient.
//   2. The leaf no longer requires a gradient (e.g. the user called
//      requires_grad_(False) after the forward pass) — discard.
//   3. Normal case: if leaf has no gradient yet, move grad_out in as-is;
//      otherwise call accumulate_into() which does an in-place += using the
//      appropriate backend (CPU element-wise loop or MLX add for GPU).
//
// Returns an empty vector because AccumulateGrad has no outgoing edges.
std::vector<Storage> AccumulateGrad::apply(Storage grad_out) {
    auto t = leaf_.lock();
    if (!t) {
        // Leaf was deallocated before backward completed; drop gradient.
        return {};
    }
    if (!t->requires_grad()) {
        return {};
    }

    auto& grad = t->mutable_grad_storage();
    if (!grad.has_value()) {
        // First gradient arriving at this leaf — take ownership directly
        // rather than allocating a zero buffer and immediately adding to it.
        grad = std::move(grad_out);
    } else {
        // Subsequent gradient: add in-place into the existing accumulator.
        accumulate_into(*grad, grad_out);
    }
    return {};
}

}  // namespace lucid
