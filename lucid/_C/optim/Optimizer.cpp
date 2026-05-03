// lucid/_C/optim/Optimizer.cpp
//
// Implementation of the abstract Optimizer base class. The two
// non-trivial methods here — step() and zero_grad() — encapsulate all
// bookkeeping that is common to every optimizer variant so that derived
// classes contain only their specific update mathematics.

#include "Optimizer.h"

#include "../core/TensorImpl.h"

namespace lucid {

// Drives one optimizer update across all registered parameters.
//
// The state_initialized_ vector is grown lazily to match params_ on
// the first step call (handles the case where params_ was extended
// after construction). A parameter is silently skipped if its pointer
// is null or if it has no gradient yet — this matches the PyTorch
// convention where parameters without gradients are treated as
// non-trainable for the current step.
void Optimizer::step() {
    if (state_initialized_.size() != params_.size()) {
        state_initialized_.assign(params_.size(), false);
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p)
            continue;
        const auto& grad = p->grad_storage();
        if (!grad.has_value())
            continue;
        if (!state_initialized_[i]) {
            init_state_slot(i, p);
            state_initialized_[i] = true;
        }
        update_one(i, p, *grad);

        // Bump the version so that any autograd nodes that captured this
        // parameter before the update detect an in-place modification.
        p->bump_version();
    }
}

// Clear accumulated gradients on all non-null parameters.
void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p)
            p->zero_grad();
    }
}

}  // namespace lucid
