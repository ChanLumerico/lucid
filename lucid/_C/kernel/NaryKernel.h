// lucid/_C/kernel/NaryKernel.h
//
// CRTP base for fixed-arity N-input, single-output ops where N >= 3.
// Compared to BinaryKernel, NaryKernel holds input metadata in
// std::array<..., N> rather than a pair, and forward() is left entirely
// to the Derived class. NaryKernel only provides a wire_autograd helper
// that Derived::forward() calls after computing the output to attach
// the backward node to the autograd graph.
//
// A typical 3-input op (e.g., linear with weight, input, and bias) calls
// NaryKernel<LinearOp, 3>::wire_autograd(inputs, out) at the end of its
// forward implementation after building the output TensorImpl.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/AutogradNode.h"
#include "../autograd/Node.h"
#include "../core/GradMode.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "BinaryKernel.h"
#include "IKernel.h"

namespace lucid {
namespace kernel {

// CRTP base for N-input ops. Inherits AutogradNode<Derived, N> to obtain
// the input_shapes_, saved_inputs_, and out_shape_ arrays at compile-time
// fixed size N. Derived provides forward() and apply() (or grad_formula()).
template <class Derived, std::size_t N>
class NaryKernel : public AutogradNode<Derived, N>, public IKernel {
public:
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    // Wire the autograd backward node bwd to out and to each input's grad_fn.
    // Returns false (no-op) when grad mode is disabled or no input requires
    // a gradient. When save_ins is true, each non-null input's storage is
    // snapshotted into saved_inputs_[i] for use during backward.
    static bool wire_autograd(std::shared_ptr<Derived> bwd,
                              const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        if (!GradMode::is_enabled())
            return false;

        bool any_grad = false;
        for (const auto& inp : inputs)
            any_grad |= (inp && inp->requires_grad());
        if (!any_grad)
            return false;

        // Derive dtype/device from the first non-null input.
        for (const auto& inp : inputs) {
            if (inp) {
                bwd->dtype_ = inp->dtype();
                bwd->device_ = inp->device();
                break;
            }
        }
        bwd->out_shape_ = out->shape();

        std::vector<Edge> edges;
        std::vector<std::int64_t> versions;
        edges.reserve(N);
        versions.reserve(N);

        for (std::size_t i = 0; i < N; ++i) {
            const auto& inp = inputs[i];
            bwd->input_shapes_[i] = inp ? inp->shape() : Shape{};
            bwd->input_tensors_[i] = inp;
            if (save_ins && inp)
                bwd->saved_inputs_[i] = inp->storage();
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

    // Convenience overload that allocates a fresh Derived backward node
    // internally rather than requiring the caller to construct one first.
    static bool wire_autograd(const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        return wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins);
    }
};

}  // namespace kernel
}  // namespace lucid
