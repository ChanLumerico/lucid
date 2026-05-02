#pragma once

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
