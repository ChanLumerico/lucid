// lucid/_C/autograd/ModuleHookNode.cpp

#include "ModuleHookNode.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <utility>

#include "../core/ErrorBuilder.h"
#include "AccumulateGrad.h"
#include "Helpers.h"
#include "TensorHooks.h"

namespace lucid {

// Declared where it is defined (ops/bfunc/Add.cpp), as Engine.cpp does: the
// ops layer sits above this one, and a create_graph sum has to be a graph op.
TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);

namespace {

py::object to_python(const TensorImplPtr& grad) {
    return grad ? py::cast(grad) : py::none();
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

// What flows on from a slot in an eager pass.  A CPU buffer is copied first:
// the hooks were handed it, or handed it back, and may keep it, while the
// engine adds into what flows on in place — a gradient a hook kept read
// ``[3, 3, 3]`` after backward instead of the ``[1, 1, 1]`` it was handed.
Storage emitted(const TensorImplPtr& grad) {
    return grad ? own_grad_copy(grad->storage()) : Storage{CpuStorage{}};
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

// One side of a hooked module as its backward hooks see it: a tuple with an
// entry per position — the module's positional arguments for a full hook,
// its tensor outputs for a pre-hook — of which only the positions a gradient
// flows through have a slot.
struct HookTuple {
    const char* op;     // the registration that installs the hooks, for errors
    const char* title;  // the hook as the reference's count error names it
    const char* noun;   // the hook in the middle of a sentence
    const char* name;   // the tuple's name: grad_input or grad_output
    // Per position: the slot a gradient replaces, or null where none flows.
    std::vector<TensorImplPtr*> slots;
    // Per position: the tensor the slot stands for, or null where none flows.
    std::vector<const ModuleHookTensorMeta*> metas;
};

HookTuple input_side(ModuleBackwardHookState& state) {
    HookTuple side{"Module.register_full_backward_hook",
                   "Backward hook",
                   "backward hook",
                   "grad_input",
                   {},
                   {}};
    side.slots.assign(state.n_inputs, nullptr);
    side.metas.assign(state.n_inputs, nullptr);
    for (std::size_t edge_idx = 0; edge_idx < state.input_arg_indices.size(); ++edge_idx) {
        const std::uint32_t pos = state.input_arg_indices[edge_idx];
        side.slots[pos] = &state.grad_inputs[edge_idx];
        side.metas[pos] = &state.input_metas[edge_idx];
    }
    return side;
}

HookTuple output_side(ModuleBackwardHookState& state) {
    HookTuple side{"Module.register_full_backward_pre_hook",
                   "Backward pre hook",
                   "backward pre-hook",
                   "grad_output",
                   {},
                   {}};
    side.slots.assign(state.n_outputs, nullptr);
    side.metas.assign(state.n_outputs, nullptr);
    for (const std::uint32_t pos : state.output_edge_indices) {
        side.slots[pos] = &state.grad_outputs[pos];
        side.metas[pos] = &state.output_metas[pos];
    }
    return side;
}

// Put a hook's ``result`` in place of the gradients ``side`` holds.
//
// ``None`` keeps them all.  Anything else is a tuple or list with one entry
// per position: ``None`` keeps that gradient, and a tensor replaces it when
// it has the kind of the gradient it replaces — the one the hook was handed,
// or where none arrived the tensor the slot stands for.  The engine reads a
// slot as that kind, so a short gradient was read past its end and a float16
// one had its bits taken for float32.  Every entry is checked before any is
// put in place.  An eager pass keeps a replacement's values in a buffer of
// the slot's form (grad_slot_buffer); a create_graph pass keeps the tensor,
// graph and all.
//
// Raises
// ------
// py::type_error
//     ``result`` is not a tuple or list, or an entry is neither a tensor nor
//     ``None``.
// LucidError
//     The count is not the tuple's, or a tensor stands where no gradient
//     flows.
// DtypeMismatch, DeviceMismatch, ShapeMismatch, NotImplementedError
//     A tensor is of another kind than the gradient it replaces, or a Metal
//     window an eager slot cannot hold.
void replace_from_hook(py::handle result, const HookTuple& side, bool graph) {
    if (result.is_none())
        return;
    const ErrorBuilder err(side.op);
    const std::string name(side.name);
    if (!py::isinstance<py::tuple>(result) && !py::isinstance<py::list>(result))
        throw py::type_error("a " + std::string(side.noun) + " must return None or a tuple of " +
                             name + ", got " +
                             std::string(py::str(py::type::of(result).attr("__name__"))));
    const auto values = py::reinterpret_borrow<py::sequence>(result);
    const std::size_t n = side.slots.size();
    if (values.size() != n)
        err.fail(std::string(side.title) + " returned an invalid number of " + name + ", got " +
                 std::to_string(values.size()) + ", but expected " + std::to_string(n));

    std::vector<TensorImplPtr> replaced(n);
    for (std::size_t pos = 0; pos < n; ++pos) {
        const std::string what =
            name + "[" + std::to_string(pos) + "] returned by a " + std::string(side.noun);
        const py::object value = values[pos];
        TensorImplPtr g = tensor_from_python(value, what);
        if (!g)
            continue;
        if (side.slots[pos] == nullptr)
            err.fail(what + " is a gradient where none flows — the value there takes no "
                            "gradient; return None in its place");
        const TensorImplPtr& current = *side.slots[pos];
        const ModuleHookTensorMeta& meta = *side.metas[pos];
        check_grad_kind(err, what, current ? current->dtype() : meta.dtype,
                        current ? current->device() : meta.device,
                        current ? current->shape() : meta.shape, *g);
        replaced[pos] = graph ? std::move(g) : as_tensor(grad_slot_buffer(err, *g), meta);
    }
    for (std::size_t pos = 0; pos < n; ++pos) {
        if (replaced[pos])
            *side.slots[pos] = std::move(replaced[pos]);
    }
}

// Run one kind of a module's hooks.  ``results`` is the Python runner's
// generator: it calls the hooks in turn and yields each result that is not
// ``None``, and each is put in place before the generator goes on, so a hook
// is never handed a tuple the engine refused and what flows on is the last
// hook's.
void run_hook_kind(const py::object& results, const HookTuple& side, bool graph) {
    for (const py::handle result : results)
        replace_from_hook(result, side, graph);
}

// Run ``body``, the hooks of one barrier; if it throws — a hook raised, or
// returned something refused — nothing any hook returned flows on: the state
// is emptied, as a finished backward leaves it, and the error goes on.
template <typename Body>
void run_or_reset(ModuleBackwardHookState& state, Body&& body) {
    try {
        body();
    } catch (...) {
        state.reset();
        throw;
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

void ModuleOutputHookNode::run_hooks(bool graph) {
    state_->enter_pass();
    py::gil_scoped_acquire gil;
    run_or_reset(*state_, [&] {
        if (!state_->pre_hooks_ran && !state_->pre_runner.is_none()) {
            run_hook_kind(state_->pre_runner(output_tuple(*state_)), output_side(*state_), graph);
            state_->pre_hooks_ran = true;
        }
        // Every position of grad_input is None here, and has to stay None.
        if (state_->input_arg_indices.empty() && !state_->full_hooks_ran &&
            !state_->full_runner.is_none()) {
            run_hook_kind(state_->full_runner(input_tuple(*state_), output_tuple(*state_)),
                          input_side(*state_), graph);
            state_->full_hooks_ran = true;
        }
    });
}

std::vector<Storage> ModuleOutputHookNode::apply_barrier() {
    if (!state_) {
        return {};
    }
    run_hooks(/*graph=*/false);
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
    run_hooks(/*graph=*/true);
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

void ModuleInputHookNode::run_hooks(bool graph) {
    state_->enter_pass();
    py::gil_scoped_acquire gil;
    run_or_reset(*state_, [&] {
        if (!state_->full_hooks_ran && !state_->full_runner.is_none()) {
            run_hook_kind(state_->full_runner(input_tuple(*state_), output_tuple(*state_)),
                          input_side(*state_), graph);
            state_->full_hooks_ran = true;
        }
    });
}

std::vector<Storage> ModuleInputHookNode::apply_barrier() {
    if (!state_) {
        return {};
    }
    run_hooks(/*graph=*/false);
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
    run_hooks(/*graph=*/true);
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
                if (arg_idx >= state->n_inputs) {
                    ErrorBuilder("_wrap_module_backward_inputs")
                        .index_error("input index out of range");
                }
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
