// lucid/_C/autograd/ModuleHookNode.cpp

#include "ModuleHookNode.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <utility>

#include "../core/ErrorBuilder.h"
#include "AccumulateGrad.h"

namespace lucid {

namespace {

py::object storage_to_python(const std::optional<Storage>& s, const ModuleHookTensorMeta& meta) {
    if (!s.has_value()) {
        return py::none();
    }
    auto impl = std::make_shared<TensorImpl>(*s, meta.shape, meta.dtype, meta.device, false);
    return py::cast(impl);
}

Storage extract_storage(py::handle obj) {
    if (obj.is_none()) {
        return Storage{CpuStorage{}};
    }
    try {
        auto impl = obj.cast<std::shared_ptr<TensorImpl>>();
        if (impl) {
            return impl->storage();
        }
    } catch (...) {
    }
    try {
        auto impl = obj.attr("impl").cast<std::shared_ptr<TensorImpl>>();
        if (impl) {
            return impl->storage();
        }
    } catch (...) {
    }
    return Storage{CpuStorage{}};
}

py::tuple make_output_tuple(const std::shared_ptr<ModuleBackwardHookState>& state) {
    py::tuple tup(state->n_outputs);
    for (std::size_t i = 0; i < state->n_outputs; ++i) {
        tup[i] = storage_to_python(state->grad_outputs[i], state->output_metas[i]);
    }
    return tup;
}

py::tuple make_input_tuple(const std::shared_ptr<ModuleBackwardHookState>& state) {
    py::tuple tup(state->n_inputs);
    for (std::size_t i = 0; i < state->n_inputs; ++i) {
        tup[i] = py::none();
    }
    for (std::size_t edge_idx = 0; edge_idx < state->input_arg_indices.size(); ++edge_idx) {
        const auto arg_idx = state->input_arg_indices[edge_idx];
        tup[arg_idx] = storage_to_python(state->grad_inputs[edge_idx], state->input_metas[edge_idx]);
    }
    return tup;
}

void apply_tuple_result_to_outputs(const py::object& result,
                                   const std::shared_ptr<ModuleBackwardHookState>& state) {
    if (result.is_none()) {
        return;
    }
    if (!py::isinstance<py::tuple>(result) && !py::isinstance<py::list>(result)) {
        return;
    }
    std::size_t idx = 0;
    for (auto item : result) {
        if (idx >= state->grad_outputs.size()) {
            break;
        }
        if (!item.is_none()) {
            state->grad_outputs[idx] = extract_storage(item);
        }
        ++idx;
    }
}

void apply_tuple_result_to_inputs(const py::object& result,
                                  const std::shared_ptr<ModuleBackwardHookState>& state) {
    if (result.is_none()) {
        return;
    }
    if (!py::isinstance<py::tuple>(result) && !py::isinstance<py::list>(result)) {
        return;
    }
    std::vector<std::optional<Storage>> by_arg(state->n_inputs);
    std::size_t arg_idx = 0;
    for (auto item : result) {
        if (arg_idx >= by_arg.size()) {
            break;
        }
        if (!item.is_none()) {
            by_arg[arg_idx] = extract_storage(item);
        }
        ++arg_idx;
    }
    for (std::size_t edge_idx = 0; edge_idx < state->input_arg_indices.size(); ++edge_idx) {
        const auto input_arg_idx = state->input_arg_indices[edge_idx];
        if (input_arg_idx < by_arg.size() && by_arg[input_arg_idx].has_value()) {
            state->grad_inputs[edge_idx] = std::move(by_arg[input_arg_idx]);
        }
    }
}

Edge edge_for(const std::shared_ptr<TensorImpl>& t) {
    if (!t || !t->requires_grad()) {
        return Edge{};
    }
    if (t->is_leaf() && !t->grad_fn()) {
        t->set_grad_fn(std::make_shared<AccumulateGrad>(t));
    }
    return Edge(t->grad_fn(), t->grad_output_nr());
}

std::shared_ptr<TensorImpl> alias_with_hook(const std::shared_ptr<TensorImpl>& base,
                                            const std::shared_ptr<Node>& node,
                                            std::uint32_t slot) {
    auto view = TensorImpl::make_view(base, base->shape(), base->stride(), 0);
    view->set_grad_fn(node);
    view->set_grad_output_nr(slot);
    view->set_leaf(false);
    view->set_requires_grad(true);
    return view;
}

}  // namespace

ModuleBackwardHookState::ModuleBackwardHookState(std::size_t n, py::object pre, py::object full)
    : pre_runner(std::move(pre)), full_runner(std::move(full)), n_inputs(n) {}

ModuleOutputHookNode::ModuleOutputHookNode(std::shared_ptr<ModuleBackwardHookState> state)
    : state_(std::move(state)) {}

std::vector<Storage> ModuleOutputHookNode::apply(Storage grad_out) {
    accumulate_barrier_grad(0, std::move(grad_out));
    return apply_barrier();
}

void ModuleOutputHookNode::accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) {
    if (!state_ || input_nr >= state_->grad_outputs.size()) {
        return;
    }
    state_->grad_outputs[input_nr] = std::move(grad);
}

std::vector<Storage> ModuleOutputHookNode::apply_barrier() {
    if (!state_) {
        return {};
    }
    py::gil_scoped_acquire gil;
    if (!state_->pre_hooks_ran && !state_->pre_runner.is_none()) {
        auto result = state_->pre_runner(make_output_tuple(state_));
        apply_tuple_result_to_outputs(result, state_);
        state_->pre_hooks_ran = true;
    }
    if (state_->input_arg_indices.empty() && !state_->full_hooks_ran && !state_->full_runner.is_none()) {
        auto result = state_->full_runner(py::tuple(0), make_output_tuple(state_));
        (void)result;
        state_->full_hooks_ran = true;
    }
    std::vector<Storage> out;
    out.reserve(state_->output_edge_indices.size());
    for (const auto out_idx : state_->output_edge_indices) {
        auto& grad = state_->grad_outputs[out_idx];
        out.push_back(grad.has_value() ? *grad : Storage{CpuStorage{}});
    }
    return out;
}

ModuleInputHookNode::ModuleInputHookNode(std::shared_ptr<ModuleBackwardHookState> state)
    : state_(std::move(state)) {}

std::vector<Storage> ModuleInputHookNode::apply(Storage grad_out) {
    accumulate_barrier_grad(0, std::move(grad_out));
    return apply_barrier();
}

void ModuleInputHookNode::accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) {
    if (!state_ || input_nr >= state_->grad_inputs.size()) {
        return;
    }
    state_->grad_inputs[input_nr] = std::move(grad);
}

std::vector<Storage> ModuleInputHookNode::apply_barrier() {
    if (!state_) {
        return {};
    }
    py::gil_scoped_acquire gil;
    if (!state_->full_hooks_ran && !state_->full_runner.is_none()) {
        auto result = state_->full_runner(make_input_tuple(state_), make_output_tuple(state_));
        apply_tuple_result_to_inputs(result, state_);
        state_->full_hooks_ran = true;
    }
    std::vector<Storage> out;
    out.reserve(state_->grad_inputs.size());
    for (auto& grad : state_->grad_inputs) {
        out.push_back(grad.has_value() ? *grad : Storage{CpuStorage{}});
    }
    return out;
}

void register_module_hook_nodes(py::module_& m) {
    py::class_<ModuleBackwardHookState, std::shared_ptr<ModuleBackwardHookState>>(
        m, "_ModuleBackwardHookState");

    m.def(
        "_create_module_backward_hook_state",
        [](std::size_t n_inputs, py::object pre_runner, py::object full_runner) {
            return std::make_shared<ModuleBackwardHookState>(
                n_inputs, std::move(pre_runner), std::move(full_runner));
        },
        py::arg("n_inputs"), py::arg("pre_runner"), py::arg("full_runner"));

    m.def(
        "_wrap_module_backward_inputs",
        [](const std::shared_ptr<ModuleBackwardHookState>& state,
           const std::vector<std::pair<std::uint32_t, std::shared_ptr<TensorImpl>>>& inputs) {
            if (!state) {
                ErrorBuilder("_wrap_module_backward_inputs").fail("state is null");
            }
            auto node = std::make_shared<ModuleInputHookNode>(state);
            std::vector<Edge> edges;
            std::vector<std::shared_ptr<TensorImpl>> wrapped;
            edges.reserve(inputs.size());
            wrapped.reserve(inputs.size());
            state->input_arg_indices.clear();
            state->input_metas.clear();
            state->grad_inputs.clear();

            for (const auto& [arg_idx, impl] : inputs) {
                const std::uint32_t slot = static_cast<std::uint32_t>(edges.size());
                edges.push_back(edge_for(impl));
                state->input_arg_indices.push_back(arg_idx);
                state->input_metas.push_back({impl->shape(), impl->dtype(), impl->device()});
                state->grad_inputs.emplace_back(std::nullopt);
                wrapped.push_back(alias_with_hook(impl, node, slot));
            }
            node->set_next_edges(std::move(edges));
            return wrapped;
        },
        py::arg("state"), py::arg("inputs"));

    m.def(
        "_wrap_module_backward_outputs",
        [](const std::shared_ptr<ModuleBackwardHookState>& state,
           const std::vector<std::pair<std::uint32_t, std::shared_ptr<TensorImpl>>>& outputs,
           std::size_t n_outputs) {
            if (!state) {
                ErrorBuilder("_wrap_module_backward_outputs").fail("state is null");
            }
            auto node = std::make_shared<ModuleOutputHookNode>(state);
            std::vector<Edge> edges;
            std::vector<std::shared_ptr<TensorImpl>> wrapped;
            edges.reserve(outputs.size());
            wrapped.reserve(outputs.size());

            state->n_outputs = n_outputs;
            state->output_metas.assign(n_outputs, {});
            state->grad_outputs.assign(n_outputs, std::nullopt);
            state->output_edge_indices.clear();
            state->output_edge_indices.reserve(outputs.size());

            for (const auto& [out_idx, impl] : outputs) {
                if (out_idx >= n_outputs) {
                    ErrorBuilder("_wrap_module_backward_outputs").fail("output index out of range");
                }
                edges.push_back(edge_for(impl));
                state->output_metas[out_idx] = {impl->shape(), impl->dtype(), impl->device()};
                state->output_edge_indices.push_back(out_idx);
                wrapped.push_back(alias_with_hook(impl, node, out_idx));
            }
            node->set_next_edges(std::move(edges));
            return wrapped;
        },
        py::arg("state"), py::arg("outputs"), py::arg("n_outputs"));
}

}  // namespace lucid
