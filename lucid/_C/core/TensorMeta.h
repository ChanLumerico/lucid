// lucid/_C/core/TensorMeta.h
//
// Plain-data structs that carry the metadata associated with a TensorImpl.
// Separating these structs from TensorImpl itself makes the autograd layer
// and view-creation code easier to read: callers can manipulate a
// TensorMeta or AutogradMeta in isolation without touching the full tensor.

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

// Immutable geometric and type description of a tensor.
//
// TensorMeta is stored directly inside TensorImpl (not on the heap) so that
// shape/dtype/device queries are a single pointer dereference.  Views share
// the same Storage as their base tensor but carry an independent TensorMeta
// (different shape, stride, or even offset).
struct TensorMeta {
    Shape shape;
    Stride stride;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    // Returns the product of all dimension sizes, or 1 for a scalar (empty
    // shape).  Does not guard against negative dimensions — callers should
    // use shape_numel() from Shape.h when dynamic negative dims are possible.
    std::size_t numel() const noexcept {
        std::size_t n = 1;
        for (auto d : shape)
            n *= static_cast<std::size_t>(d);
        return n;
    }

    // Returns true if the stored stride vector matches the row-major
    // (C-contiguous) stride for elem_size-byte elements.  Used by
    // is_contiguous() in TensorImpl and by op kernels that require
    // contiguous input.
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

// Autograd bookkeeping for a single tensor.
//
// This struct is stored as an std::optional<AutogradMeta> inside TensorImpl
// and is only heap-constructed when a tensor actually participates in the
// autograd graph (requires_grad == true or a grad_fn is attached).  Tensors
// that never need gradients pay zero overhead for the optional.
//
// Invariants:
//   - is_leaf == true for tensors created directly by the user (no grad_fn).
//   - is_leaf == false for outputs of differentiable operations; such tensors
//     always have a non-null grad_fn.
//   - grad holds the accumulated gradient Storage after backward(); it is
//     cleared by zero_grad().
//   - version is bumped on every in-place write via TensorImpl::bump_version();
//     autograd saves the pre-forward version and raises VersionMismatch if the
//     tensor is mutated between forward and backward.
struct AutogradMeta {
    bool requires_grad = false;
    bool is_leaf = true;
    std::int64_t version = 0;
    std::shared_ptr<Node> grad_fn;
    std::uint32_t grad_output_nr = 0;
    // Accumulated gradient Storage; set on leaves after normal backward().
    std::optional<Storage> grad;
    // Gradient as a full TensorImpl when backward was run with create_graph=true.
    // Allows the gradient tensor itself to participate in further autograd ops
    // (second-order derivatives, MAML, Hessian-vector products, etc.).
    std::shared_ptr<class TensorImpl> grad_impl;
    // When true, Engine accumulates the incoming gradient into this tensor's
    // grad storage even if it is not a leaf (mirrors reference tensor.retain_grad()).
    bool retain_grad = false;
};

}  // namespace lucid
