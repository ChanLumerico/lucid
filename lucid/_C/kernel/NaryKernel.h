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

template <class Derived, std::size_t N>
class NaryKernel : public AutogradNode<Derived, N>, public IKernel {
public:
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

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

    static bool wire_autograd(const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        return wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins);
    }
};

}  // namespace kernel
}  // namespace lucid
