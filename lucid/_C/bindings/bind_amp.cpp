#include <pybind11/pybind11.h>

#include "../core/AmpPolicy.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_amp(py::module_& m) {
    py::enum_<AmpPolicy>(m, "AmpPolicy")
        .value("Promote", AmpPolicy::Promote)
        .value("KeepInput", AmpPolicy::KeepInput)
        .value("ForceFP32", AmpPolicy::ForceFP32)
        .export_values();
    m.def("amp_policy_name", [](AmpPolicy p) { return std::string(amp_policy_name(p)); });

    py::class_<amp::AutocastGuard>(m, "AutocastGuard")
        .def(py::init<Dtype>(), py::arg("target"))
        .def(
            "__enter__", [](amp::AutocastGuard& g) -> amp::AutocastGuard& { return g; },
            py::return_value_policy::reference)
        .def("__exit__",
             [](amp::AutocastGuard&, py::object, py::object, py::object) { return py::none(); });

    m.def("amp_active_dtype", []() -> py::object {
        auto d = amp::active_dtype();
        if (d)
            return py::cast(*d);
        return py::none();
    });
    m.def("amp_is_active", &amp::is_active);
}

}  // namespace lucid::bindings
