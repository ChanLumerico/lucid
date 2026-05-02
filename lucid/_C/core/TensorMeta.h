#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "Device.h"
#include "Dtype.h"
#include "Shape.h"
#include "Storage.h"

namespace lucid {

class Node;

struct TensorMeta {
    Shape shape;
    Stride stride;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    std::size_t numel() const noexcept {
        std::size_t n = 1;
        for (auto d : shape)
            n *= static_cast<std::size_t>(d);
        return n;
    }

    bool is_contiguous_for(std::size_t elem_size) const noexcept {
        if (shape.empty())
            return true;
        if (stride.size() != shape.size())
            return false;
        std::int64_t expected = static_cast<std::int64_t>(elem_size);
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            if (stride[static_cast<std::size_t>(i)] != expected)
                return false;
            expected *= shape[static_cast<std::size_t>(i)];
        }
        return true;
    }
};

struct AutogradMeta {
    bool requires_grad = false;
    bool is_leaf = true;
    std::int64_t version = 0;
    std::shared_ptr<Node> grad_fn;
    std::optional<Storage> grad;
};

}  // namespace lucid
