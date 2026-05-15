// lucid/_C/autograd/ModuleHookNode.h
//
// Backward hook barriers used by nn.Module. They are only inserted when a
// module has backward hooks, keeping the normal autograd path untouched.

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/TensorImpl.h"
#include "Node.h"

namespace py = pybind11;

namespace lucid {

struct ModuleHookTensorMeta {
    Shape shape;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;
};

class ModuleBackwardHookState : public std::enable_shared_from_this<ModuleBackwardHookState> {
public:
    ModuleBackwardHookState(std::size_t n_inputs, py::object pre_runner, py::object full_runner);

    py::object pre_runner;
    py::object full_runner;
    std::size_t n_inputs = 0;
    std::size_t n_outputs = 0;
    bool pre_hooks_ran = false;
    bool full_hooks_ran = false;

    std::vector<std::uint32_t> input_arg_indices;
    std::vector<ModuleHookTensorMeta> input_metas;
    std::vector<std::optional<Storage>> grad_inputs;

    std::vector<ModuleHookTensorMeta> output_metas;
    std::vector<std::optional<Storage>> grad_outputs;
    std::vector<std::uint32_t> output_edge_indices;
};

class ModuleOutputHookNode : public Node {
public:
    explicit ModuleOutputHookNode(std::shared_ptr<ModuleBackwardHookState> state);

    std::vector<Storage> apply(Storage grad_out) override;
    bool is_barrier() const noexcept override { return true; }
    void accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) override;
    std::vector<Storage> apply_barrier() override;
    std::string node_name() const override { return "ModuleOutputHook"; }

private:
    std::shared_ptr<ModuleBackwardHookState> state_;
};

class ModuleInputHookNode : public Node {
public:
    explicit ModuleInputHookNode(std::shared_ptr<ModuleBackwardHookState> state);

    std::vector<Storage> apply(Storage grad_out) override;
    bool is_barrier() const noexcept override { return true; }
    void accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) override;
    std::vector<Storage> apply_barrier() override;
    std::string node_name() const override { return "ModuleInputHook"; }

private:
    std::shared_ptr<ModuleBackwardHookState> state_;
};

void register_module_hook_nodes(py::module_& m);

}  // namespace lucid
