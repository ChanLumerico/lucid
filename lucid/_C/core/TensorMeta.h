#pragma once

// =====================================================================
// Lucid C++ engine — TensorMeta: shape/stride/dtype/device bundle.
// =====================================================================
//
// `TensorMeta` holds the pure-metadata portion of a tensor that is
// independent of autograd. It is embedded inside `TensorImpl` as the
// private `meta_` member and also used by view-op builders and kernels
// that need to describe an output shape/dtype without owning storage.
//
// `AutogradMeta` groups all autograd bookkeeping fields. `TensorImpl`
// stores it as `std::optional<AutogradMeta> autograd_` — the optional
// is empty for inference-only tensors, saving the Node shared_ptr cost.
//
// Layer: core/. No dependencies beyond core/Shape.h, core/Dtype.h,
//        core/Device.h, autograd/Node.h (forward-declared here).

#include <cstdint>
#include <memory>
#include <optional>

#include "Device.h"
#include "Dtype.h"
#include "Shape.h"
#include "Storage.h"

namespace lucid {

class Node;

// --------------------------------------------------------------------------- //
// TensorMeta — shape, stride, dtype, device.
// --------------------------------------------------------------------------- //

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

    // is_contiguous() requires dtype knowledge (strides are byte-strides).
    // Use TensorImpl::is_contiguous() externally; this helper is for
    // view-op builders that have the elem_size in scope.
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

// --------------------------------------------------------------------------- //
// AutogradMeta — autograd bookkeeping (optional allocation).
// --------------------------------------------------------------------------- //

struct AutogradMeta {
    bool requires_grad = false;
    bool is_leaf = true;
    std::int64_t version = 0;
    std::shared_ptr<Node> grad_fn;
    std::optional<Storage> grad;
};

}  // namespace lucid
