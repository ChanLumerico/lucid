#pragma once

#include <memory>

#include "../core/TensorImpl.h"
#include "Node.h"

namespace lucid {

class AccumulateGrad : public Node {
public:
    explicit AccumulateGrad(std::weak_ptr<TensorImpl> leaf);

    std::vector<Storage> apply(Storage grad_out) override;

    std::weak_ptr<TensorImpl> leaf() const { return leaf_; }

private:
    std::weak_ptr<TensorImpl> leaf_;
};

}  // namespace lucid
