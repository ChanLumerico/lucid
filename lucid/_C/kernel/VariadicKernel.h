// lucid/_C/kernel/VariadicKernel.h
//
// CRTP base for ops that take a runtime-variable number of inputs, the
// canonical example being concatenation (cat) which accepts any number of
// tensors along a specified dimension. Unlike NaryKernel<Derived, N>,
// which fixes N at compile time, VariadicKernel stores per-input metadata
// in std::vector rather than std::array.
//
// Because the input count is dynamic, VariadicKernel inherits directly from
// Node (not from AutogradNode<Derived, N>). It replicates the input_shapes_,
// input_tensors_, saved_inputs_, and out_shape_ pattern but in vector form
// (suffixed _v_ to avoid collisions with AutogradNode members when a Derived
// type also inherits another kernel base).
//
// A variadic op's forward() allocates its output, then calls:
//   VariadicKernel<Derived>::wire_autograd(inputs, out);

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/GradMode.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "BinaryKernel.h"
#include "IKernel.h"

namespace lucid {

class TensorImpl;

namespace kernel {

// CRTP base for variable-arity ops. Inherits Node directly rather than
// AutogradNode<N> to support a runtime input count. Derived classes must
// implement apply(Storage) for the backward pass.
template <class Derived>
class VariadicKernel : public Node, public IKernel {
public:
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    // Version-staleness check for all variadic inputs. Called by the
    // autograd engine before apply() to detect in-place mutations.
    void validate_versions() override {
        for (std::size_t i = 0; i < input_tensors_v_.size(); ++i) {
            ::lucid::check_version_match(input_tensors_v_[i],
                                         saved_versions_.size() > i ? saved_versions_[i] : 0,
                                         Derived::schema_v1.name, i);
        }
    }

    // Weak references to original input tensors; used for version checks.
    std::vector<std::weak_ptr<TensorImpl>> input_tensors_v_;
    // Shapes of the original inputs, preserved for backward shape math.
    std::vector<Shape> input_shapes_v_;
    // Shape of the produced output tensor.
    Shape out_shape_;
    // Dtype and device of the first non-null input (all inputs must match).
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;
    // Snapshots of each input's Storage at forward time, available to apply().
    std::vector<Storage> saved_inputs_v_;
    // Optional snapshot of the output storage (used by ops whose backward
    // needs the forward output value, e.g., softmax).
    Storage saved_output_;

    // Wire the provided backward node bwd to out and to each input edge.
    // Returns false if grad mode is off or no input requires a gradient.
    // When save_ins is true, each input's storage is captured into
    // saved_inputs_v_ for use in apply().
    static bool wire_autograd(std::shared_ptr<Derived> bwd,
                              const std::vector<TensorImplPtr>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        if (!GradMode::is_enabled())
            return false;

        bool any_grad = false;
        for (const auto& inp : inputs)
            any_grad |= (inp && inp->requires_grad());
        if (!any_grad)
            return false;

        // Use the first non-null input to resolve dtype and device.
        for (const auto& inp : inputs) {
            if (inp) {
                bwd->dtype_ = inp->dtype();
                bwd->device_ = inp->device();
                break;
            }
        }
        bwd->out_shape_ = out->shape();

        const std::size_t n = inputs.size();
        bwd->input_shapes_v_.reserve(n);
        bwd->input_tensors_v_.reserve(n);
        if (save_ins)
            bwd->saved_inputs_v_.reserve(n);

        std::vector<Edge> edges;
        std::vector<std::int64_t> versions;
        edges.reserve(n);
        versions.reserve(n);

        for (const auto& inp : inputs) {
            bwd->input_shapes_v_.push_back(inp ? inp->shape() : Shape{});
            bwd->input_tensors_v_.push_back(inp);
            if (save_ins && inp)
                bwd->saved_inputs_v_.push_back(inp->storage());
            edges.emplace_back(detail::ensure_grad_fn(inp), 0);
            versions.push_back(inp ? inp->version() : 0);
        }

        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions(std::move(versions));

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return true;
    }

    // Convenience overload that creates the backward node internally.
    static bool wire_autograd(const std::vector<TensorImplPtr>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        return wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins);
    }
};

}  // namespace kernel
}  // namespace lucid
