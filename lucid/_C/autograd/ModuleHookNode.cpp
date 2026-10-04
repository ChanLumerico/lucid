// lucid/_C/autograd/ModuleHookNode.cpp

#include "ModuleHookNode.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <utility>

#include "../core/ErrorBuilder.h"
#include "AccumulateGrad.h"
#include "Helpers.h"

namespace lucid {

// Declared where it is defined (ops/bfunc/Add.cpp), as Engine.cpp does: the
// ops layer sits above this one, and a create_graph sum has to be a graph op.
TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);

namespace {

py::object to_python(const TensorImplPtr& grad) {
    return grad ? py::cast(grad) : py::none();
}

// The tensor a hook returned, or null for anything that is not one.
TensorImplPtr extract_impl(py::handle obj) {
    if (obj.is_none()) {
        return nullptr;
    }
    try {
        return obj.cast<std::shared_ptr<TensorImpl>>();
    } catch (...) {
    }
    try {
        return obj.attr("impl").cast<std::shared_ptr<TensorImpl>>();
    } catch (...) {
    }
    return nullptr;
}

// An eager gradient as the tensor the hooks receive.
TensorImplPtr as_tensor(Storage grad, const ModuleHookTensorMeta& meta) {
    const Dtype dtype = storage_dtype(grad);
    return std::make_shared<TensorImpl>(std::move(grad), meta.shape, dtype, meta.device, false);
}

// ``parked + arriving``, in a buffer of its own.
//
// The parked gradient is the buffer its producer handed on, and that buffer
// can have other holders — a ``retain_grad`` slot, a sibling edge's pending
// gradient — so adding into it would change a gradient that is not this
// slot's.  A metal add already lands in a new array; a CPU one adds into the
// buffer, so it gets a copy first.
Storage summed(const Storage& parked, const Storage& arriving) {
    Storage sum = parked;
    if (storage_is_cpu(parked)) {
        const Dtype dtype = storage_dtype(parked);
        sum = clone_storage(parked, storage_nbytes(parked) / dtype_size(dtype), dtype, Device::CPU);
    }
    accumulate_into(sum, arriving);
    return sum;
}

void add_to_slot(TensorImplPtr& slot, Storage grad, const ModuleHookTensorMeta& meta) {
    slot = slot ? as_tensor(summed(slot->storage(), grad), meta) : as_tensor(std::move(grad), meta);
}

void add_to_slot_for_graph(TensorImplPtr& slot, TensorImplPtr grad) {
    slot = slot ? add_op(slot, grad) : std::move(grad);
}

Storage emitted(const TensorImplPtr& grad) {
    return grad ? grad->storage() : Storage{CpuStorage{}};
}

py::tuple output_tuple(const ModuleBackwardHookState& state) {
    py::tuple tup(state.n_outputs);
    for (std::size_t i = 0; i < state.n_outputs; ++i) {
        tup[i] = to_python(state.grad_outputs[i]);
    }
    return tup;
}

py::tuple input_tuple(const ModuleBackwardHookState& state) {
    py::tuple tup(state.n_inputs);
    for (std::size_t i = 0; i < state.n_inputs; ++i) {
        tup[i] = py::none();
    }
    for (std::size_t edge_idx = 0; edge_idx < state.input_arg_indices.size(); ++edge_idx) {
        tup[state.input_arg_indices[edge_idx]] = to_python(state.grad_inputs[edge_idx]);
    }
    return tup;
}

// A hook's returned tuple replaces the gradients it gives; ``None`` keeps one.
void replace_outputs(const py::object& result, ModuleBackwardHookState& state) {
    if (result.is_none()) {
        return;
    }
    if (!py::isinstance<py::tuple>(result) && !py::isinstance<py::list>(result)) {
        return;
    }
    std::size_t idx = 0;
    for (auto item : result) {
        if (idx >= state.grad_outputs.size()) {
            break;
        }
        if (!item.is_none()) {
            state.grad_outputs[idx] = extract_impl(item);
        }
        ++idx;
    }
}

void replace_inputs(const py::object& result, ModuleBackwardHookState& state) {
    if (result.is_none()) {
        return;
    }
    if (!py::isinstance<py::tuple>(result) && !py::isinstance<py::list>(result)) {
        return;
    }
    std::vector<std::optional<TensorImplPtr>> by_arg(state.n_inputs);
    std::size_t arg_idx = 0;
    for (auto item : result) {
        if (arg_idx >= by_arg.size()) {
            break;
        }
        if (!item.is_none()) {
            by_arg[arg_idx] = extract_impl(item);
        }
        ++arg_idx;
    }
    for (std::size_t edge_idx = 0; edge_idx < state.input_arg_indices.size(); ++edge_idx) {
        const auto input_arg_idx = state.input_arg_indices[edge_idx];
        if (input_arg_idx < by_arg.size() && by_arg[input_arg_idx].has_value()) {
            state.grad_inputs[edge_idx] = std::move(*by_arg[input_arg_idx]);
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
    // The alias stands in for the output in the graph; it is not a view the
    // user made, so it stays out of the output's view family and an in-place
    // op on it swaps its own slot, as it always has.
    auto view =
        TensorImpl::make_view(base, base->shape(), base->stride(), 0, /*join_family=*/false);
    view->set_grad_fn(node);
    view->set_grad_output_nr(slot);
    view->set_leaf(false);
    view->set_requires_grad(true);
    return view;
}

}  // namespace

ModuleBackwardHookState::ModuleBackwardHookState(std::size_t n, py::object pre, py::object full)
    : pre_runner(std::move(pre)), full_runner(std::move(full)), n_inputs(n) {}

void ModuleBackwardHookState::enter_pass() {
    const std::uint64_t current = BackwardPass::current();
    if (current == pass) {
        return;
    }
    reset();
    pass = current;
}

void ModuleBackwardHookState::reset() {
    std::fill(grad_inputs.begin(), grad_inputs.end(), nullptr);
    std::fill(grad_outputs.begin(), grad_outputs.end(), nullptr);
    pre_hooks_ran = false;
    full_hooks_ran = false;
}

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
    state_->enter_pass();
    add_to_slot(state_->grad_outputs[input_nr], std::move(grad), state_->output_metas[input_nr]);
}

void ModuleOutputHookNode::accumulate_barrier_grad_for_graph(std::uint32_t input_nr,
                                                             TensorImplPtr grad) {
    if (!state_ || !grad || input_nr >= state_->grad_outputs.size()) {
        return;
    }
    state_->enter_pass();
    add_to_slot_for_graph(state_->grad_outputs[input_nr], std::move(grad));
}

void ModuleOutputHookNode::run_hooks() {
    state_->enter_pass();
    py::gil_scoped_acquire gil;
    if (!state_->pre_hooks_ran && !state_->pre_runner.is_none()) {
        replace_outputs(state_->pre_runner(output_tuple(*state_)), *state_);
        state_->pre_hooks_ran = true;
    }
    if (state_->input_arg_indices.empty() && !state_->full_hooks_ran &&
        !state_->full_runner.is_none()) {
        state_->full_runner(py::tuple(0), output_tuple(*state_));
        state_->full_hooks_ran = true;
    }
}

std::vector<Storage> ModuleOutputHookNode::apply_barrier() {
    if (!state_) {
        return {};
    }
    run_hooks();
    std::vector<Storage> out;
    out.reserve(state_->output_edge_indices.size());
    for (const auto out_idx : state_->output_edge_indices) {
        out.push_back(emitted(state_->grad_outputs[out_idx]));
    }
    // With no input that takes a gradient, the module's backward ends here.
    if (state_->input_arg_indices.empty()) {
        state_->reset();
    }
    return out;
}

std::vector<TensorImplPtr> ModuleOutputHookNode::apply_barrier_for_graph() {
    if (!state_) {
        return {};
    }
    run_hooks();
    std::vector<TensorImplPtr> out;
    out.reserve(state_->output_edge_indices.size());
    for (const auto out_idx : state_->output_edge_indices) {
        out.push_back(state_->grad_outputs[out_idx]);
    }
    if (state_->input_arg_indices.empty()) {
        state_->reset();
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
    state_->enter_pass();
    add_to_slot(state_->grad_inputs[input_nr], std::move(grad), state_->input_metas[input_nr]);
}

void ModuleInputHookNode::accumulate_barrier_grad_for_graph(std::uint32_t input_nr,
                                                            TensorImplPtr grad) {
    if (!state_ || !grad || input_nr >= state_->grad_inputs.size()) {
        return;
    }
    state_->enter_pass();
    add_to_slot_for_graph(state_->grad_inputs[input_nr], std::move(grad));
}

void ModuleInputHookNode::run_hooks() {
    state_->enter_pass();
    py::gil_scoped_acquire gil;
    if (!state_->full_hooks_ran && !state_->full_runner.is_none()) {
        replace_inputs(state_->full_runner(input_tuple(*state_), output_tuple(*state_)), *state_);
        state_->full_hooks_ran = true;
    }
}

std::vector<Storage> ModuleInputHookNode::apply_barrier() {
    if (!state_) {
        return {};
    }
    run_hooks();
    std::vector<Storage> out;
    out.reserve(state_->grad_inputs.size());
    for (const auto& grad : state_->grad_inputs) {
        out.push_back(emitted(grad));
    }
    state_->reset();
    return out;
}

std::vector<TensorImplPtr> ModuleInputHookNode::apply_barrier_for_graph() {
    if (!state_) {
        return {};
    }
    run_hooks();
    std::vector<TensorImplPtr> out = state_->grad_inputs;
    state_->reset();
    return out;
}

void register_module_hook_nodes(py::module_& m) {
    py::class_<ModuleBackwardHookState, std::shared_ptr<ModuleBackwardHookState>>(
        m, "_ModuleBackwardHookState");

    m.def(
        "_create_module_backward_hook_state",
        [](std::size_t n_inputs, py::object pre_runner, py::object full_runner) {
            return std::make_shared<ModuleBackwardHookState>(n_inputs, std::move(pre_runner),
                                                             std::move(full_runner));
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
                state->grad_inputs.emplace_back(nullptr);
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
            state->grad_outputs.assign(n_outputs, nullptr);
            state->output_edge_indices.clear();
            state->output_edge_indices.reserve(outputs.size());

            for (const auto& [out_idx, impl] : outputs) {
                if (out_idx >= n_outputs) {
                    ErrorBuilder("_wrap_module_backward_outputs")
                        .index_error("output index out of range");
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
