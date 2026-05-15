// lucid/_C/bindings/bind_amp.cpp
//
// Exposes the Automatic Mixed Precision (AMP) subsystem to Python.  Registers:
//   - AmpPolicy enum (Promote, KeepInput, ForceFP32)
//   - amp_policy_name() string helper
//   - AutocastGuard context manager (wraps amp::AutocastGuard)
//   - amp_active_dtype() / amp_is_active() query functions
//
// Python usage: `with engine.AutocastGuard(engine.Dtype.F16): ...`

#include <pybind11/pybind11.h>

#include "../core/AmpPolicy.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers the AmpPolicy enum, the AutocastGuard context manager, and the
// AMP query helpers.
void register_amp(py::module_& m) {
    // AmpPolicy controls how dtype promotion is applied inside an autocast
    // scope.  export_values() lets callers use engine.Promote etc. directly.
    py::enum_<AmpPolicy>(m, "AmpPolicy")
        .value("Promote", AmpPolicy::Promote)
        .value("KeepInput", AmpPolicy::KeepInput)
        .value("ForceFP32", AmpPolicy::ForceFP32)
        .export_values();
    // amp_policy_name converts the enum to a human-readable string for repr
    // and error messages; the lambda materialises the string_view as std::string.
    m.def("amp_policy_name", [](AmpPolicy p) { return std::string(amp_policy_name(p)); });

    // AutocastGuard is a RAII scope that pushes a target Dtype onto the AMP
    // stack on construction and pops it on destruction.  The context-manager
    // protocol is implemented by __enter__ / __exit__; __enter__ returns a
    // reference to self because Python's `with` statement binds the result but
    // ownership stays with the outer scope.
    py::class_<amp::AutocastGuard>(m, "AutocastGuard")
        .def(py::init<Dtype>(), py::arg("target"))
        .def(
            "__enter__", [](amp::AutocastGuard& g) -> amp::AutocastGuard& { return g; },
            py::return_value_policy::reference)
        .def("__exit__",
             [](amp::AutocastGuard&, py::object, py::object, py::object) { return py::none(); });

    // amp_active_dtype returns None when no autocast scope is active, or the
    // current target Dtype when one is.  The optional is unpacked here rather
    // than relying on pybind11's optional support to keep the None path explicit.
    m.def("amp_active_dtype", []() -> py::object {
        auto d = amp::active_dtype();
        if (d)
            return py::cast(*d);
        return py::none();
    });
    m.def("amp_is_active", &amp::is_active);
}

}  // namespace lucid::bindings
