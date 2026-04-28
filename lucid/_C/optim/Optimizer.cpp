#include "Optimizer.h"

#include "../core/TensorImpl.h"

namespace lucid {

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
            continue;  // no grad: skip
        if (!state_initialized_[i]) {
            init_state_slot(i, p);
            state_initialized_[i] = true;
        }
        update_one(i, p, *grad);
        // Item #9 — version bump invalidates any saved_inputs that
        // captured this parameter, so a stale backward will fail loudly.
        p->bump_version();
    }
}

void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p)
            p->zero_grad();
    }
}

}  // namespace lucid
