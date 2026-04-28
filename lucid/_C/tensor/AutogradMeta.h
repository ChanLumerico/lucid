#pragma once

// =====================================================================
// Lucid C++ engine — TensorImpl autograd metadata.
// =====================================================================
//
// Kept separate from TensorMeta so Phase 2 can make non-grad tensors cheap and
// Phase 2.5 can move versioning onto shared storage without changing the
// public TensorImpl accessors again.

#include <cstdint>
#include <optional>

#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

struct AutogradMeta {
    bool requires_grad = false;
    bool is_leaf = true;
    std::int64_t version = 0;
    NodePtr grad_fn;
    std::optional<Storage> grad;
};

}  // namespace lucid
