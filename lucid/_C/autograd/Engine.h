#pragma once

#include <memory>

#include "../api.h"
#include "../core/TensorImpl.h"

namespace lucid {

class LUCID_API Engine {
public:
    static void backward(const std::shared_ptr<TensorImpl>& root,
                         Storage grad_seed = Storage{CpuStorage{}},
                         bool retain_graph = false);
};

}  // namespace lucid
