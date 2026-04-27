#pragma once

#include <memory>

#include "../core/TensorImpl.h"
#include "Node.h"

namespace lucid {

// Sentinel Node for leaf tensors. Never has next_edges. Its `apply()` writes
// the incoming gradient into the leaf's `grad_storage_`, accumulating with any
// existing gradient.
class AccumulateGrad : public Node {
public:
    explicit AccumulateGrad(std::weak_ptr<TensorImpl> leaf);

    std::vector<Storage> apply(Storage grad_out) override;

    std::weak_ptr<TensorImpl> leaf() const { return leaf_; }

private:
    std::weak_ptr<TensorImpl> leaf_;
};

}  // namespace lucid
