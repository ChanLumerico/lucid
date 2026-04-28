#include "AccumulateGrad.h"

#include <utility>

#include "Helpers.h"

namespace lucid {

AccumulateGrad::AccumulateGrad(std::weak_ptr<TensorImpl> leaf) : leaf_(std::move(leaf)) {}

std::vector<Storage> AccumulateGrad::apply(Storage grad_out) {
    auto t = leaf_.lock();
    if (!t) {
        // Leaf was freed before backward finished. Silently drop.
        return {};
    }
    if (!t->requires_grad()) {
        return {};
    }

    auto& grad = t->mutable_grad_storage();
    if (!grad.has_value()) {
        grad = std::move(grad_out);
    } else {
        accumulate_into(*grad, grad_out);
    }
    return {};
}

}  // namespace lucid
