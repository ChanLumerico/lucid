// lucid/_C/bindings/bind_amp.cpp
//
// Exposes the Automatic Mixed Precision (AMP) subsystem to Python.  Registers:
//   - AmpPolicy enum (Promote, KeepInput, ForceFP32, PromoteWidest)
//   - amp_policy_for() / amp_policy_table() — the per-device policy table
//   - amp_cast_inputs() — a Python composite's row, applied to its inputs
//   - AutocastGuard context manager (wraps amp::AutocastGuard)
//   - amp_active_dtype() / amp_is_active() query functions
//
// Python usage: `with engine.AutocastGuard(engine.Dtype.F16): ...`

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../core/AmpPolicy.h"
#include "../core/TensorImpl.h"
#include "../ops/utils/Promote.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers the AmpPolicy enum, the policy table, the AutocastGuard context
// manager, and the AMP query helpers.
void register_amp(py::module_& m) {
    // AmpPolicy controls how dtype promotion is applied inside an autocast
    // scope.  export_values() lets callers use engine.Promote etc. directly.
    py::enum_<AmpPolicy>(m, "AmpPolicy")
        .value("Promote", AmpPolicy::Promote)
        .value("KeepInput", AmpPolicy::KeepInput)
        .value("ForceFP32", AmpPolicy::ForceFP32)
        .value("PromoteWidest", AmpPolicy::PromoteWidest)
        .export_values();

    // The table the engine applies, so Python composites and tooling read
    // the same rows rather than a copy of them.
    m.def(
        "amp_policy_for",
        [](const std::string& name, Device device) { return amp::policy_for(name, device); },
        py::arg("name"), py::arg("device"));
    m.def("amp_policy_table", []() {
        std::vector<std::tuple<std::string, AmpPolicy, AmpPolicy>> rows;
        for (const auto& row : amp::policy_table())
            rows.emplace_back(std::string(row.name), row.metal, row.cpu);
        return rows;
    });
    // A Python composite's autocast entry: its row applied to its floating
    // inputs by the engine's one owner of the cast rule (AmpScope), so the
    // rule is not written out again in Python.  The caller turns autocast
    // off for the composite's body.  Outside autocast the inputs come back
    // as they went in.  The first present input selects the device, as
    // promote_for_schema's does; absent ones (None) pass through.
    m.def(
        "amp_cast_inputs",
        [](const std::string& name, std::vector<TensorImplPtr> inputs) {
            const auto first = std::find_if(inputs.begin(), inputs.end(),
                                            [](const TensorImplPtr& t) { return t != nullptr; });
            if (first == inputs.end())
                return inputs;
            const AmpScope amp{name, (*first)->device()};
            return amp.cast_widest(std::move(inputs));
        },
        py::arg("name"), py::arg("inputs"));

    // AutocastGuard is a RAII scope that pushes a target Dtype onto the AMP
    // stack on construction and pops it on destruction.  The context-manager
    // protocol is implemented by __enter__ / __exit__; __enter__ returns a
    // reference to self because Python's `with` statement binds the result but
    // ownership stays with the outer scope.
    py::class_<amp::AutocastGuard>(m, "AutocastGuard")
        .def(py::init<Dtype>(), py::arg("target"))
        .def(py::init<Dtype, Device, bool>(), py::arg("target"), py::arg("device"),
             py::arg("enabled") = true)
        .def(
            "__enter__", [](amp::AutocastGuard& g) -> amp::AutocastGuard& { return g; },
            py::return_value_policy::reference)
        .def("__exit__",
             [](amp::AutocastGuard&, py::object, py::object, py::object) { return py::none(); });

    // amp_active_dtype returns None when no autocast scope is active, or the
    // current target Dtype when one is.  The optional is unpacked here rather
    // than relying on pybind11's optional support to keep the None path explicit.
    m.def(
        "amp_active_dtype",
        [](py::object device) -> py::object {
            auto d =
                device.is_none() ? amp::active_dtype() : amp::active_dtype(device.cast<Device>());
            if (d)
                return py::cast(*d);
            return py::none();
        },
        py::arg("device") = py::none());
    m.def(
        "amp_is_active",
        [](py::object device) -> bool {
            return device.is_none() ? amp::is_active() : amp::is_active(device.cast<Device>());
        },
        py::arg("device") = py::none());
}

}  // namespace lucid::bindings
