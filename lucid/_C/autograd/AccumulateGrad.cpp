// lucid/_C/autograd/AccumulateGrad.cpp
//
// Implements AccumulateGrad::apply(), which is the final gradient sink for
// every leaf tensor that participates in a backward pass.

#include "AccumulateGrad.h"

#include <utility>

#include "../backend/Dispatcher.h"
#include "../core/Storage.h"
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

    // 3.3 AMP fix: under autocast, the same leaf parameter can be reached
    // via two different effective dtypes — e.g. a Conv with eff_dt=F16
    // emits an F16 grad while a sibling path that ran ForceFP32 emits an
    // F32 grad.  ``accumulate_into`` asserts identical dtype on GPU and
    // would throw DtypeMismatch in that case.  Always cast incoming grads
    // to the leaf parameter's own dtype before storing/accumulating —
    // this matches the reference framework's policy of keeping the
    // gradient slot at the parameter's dtype.
    const Dtype target_dt = t->dtype();
    const Dtype src_dt = storage_dtype(grad_out);
    if (src_dt != target_dt) {
        auto& be = backend::Dispatcher::for_device(t->device());
        grad_out = be.astype(grad_out, t->shape(), src_dt, target_dt);
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

// Store grad_out (a TensorImplPtr with its own grad_fn) into the leaf's
// grad_impl slot so the gradient tensor itself is differentiable.
// This path is taken when Engine::backward is called with create_graph=true.
std::vector<TensorImplPtr> AccumulateGrad::apply_for_graph(const TensorImplPtr& grad_out) {
    auto t = leaf_.lock();
    if (!t || !t->requires_grad()) {
        return {};
    }
    t->accumulate_grad_impl(grad_out);
    return {};
}

}  // namespace lucid
