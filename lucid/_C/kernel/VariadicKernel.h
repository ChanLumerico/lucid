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

template <class Derived>
class VariadicKernel : public Node, public IKernel {
public:
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    void validate_versions() override {
        for (std::size_t i = 0; i < input_tensors_v_.size(); ++i) {
            ::lucid::check_version_match(input_tensors_v_[i],
                                         saved_versions_.size() > i ? saved_versions_[i] : 0,
                                         Derived::schema_v1.name, i);
        }
    }

    std::vector<std::weak_ptr<TensorImpl>> input_tensors_v_;
    std::vector<Shape> input_shapes_v_;
    Shape out_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;
    std::vector<Storage> saved_inputs_v_;
    Storage saved_output_;

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

    static bool wire_autograd(const std::vector<TensorImplPtr>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        return wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins);
    }
};

}  // namespace kernel
}  // namespace lucid
