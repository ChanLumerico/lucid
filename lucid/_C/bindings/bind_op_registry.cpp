#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/OpRegistry.h"
#include "../core/OpSchema.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_op_registry(py::module_& m) {
    py::class_<OpSchema>(m, "OpSchema")
        .def_property_readonly("name", [](const OpSchema& s) { return std::string(s.name); })
        .def_readonly("version", &OpSchema::version)
        .def_readonly("amp_policy", &OpSchema::amp_policy)
        .def_readonly("deterministic", &OpSchema::deterministic)
        .def_property_readonly("determinism_note",
                               [](const OpSchema& s) { return std::string(s.determinism_note); })
        .def_readonly("input_arity", &OpSchema::input_arity)
        .def_readonly("output_arity", &OpSchema::output_arity)
        .def_readonly("stable_input_indices", &OpSchema::stable_input_indices)
        .def_readonly("internal", &OpSchema::internal)
        .def("__repr__", [](const OpSchema& s) {
            return "OpSchema(name='" + std::string(s.name) +
                   "', version=" + std::to_string(s.version) +
                   ", input_arity=" + std::to_string(s.input_arity) + ")";
        });

    m.def("schema_hash", &schema_hash, py::arg("schema"));

    m.def(
        "op_lookup",
        [](std::string_view name) -> py::object {
            const auto* s = OpRegistry::lookup(name);
            if (!s)
                return py::none();
            return py::cast(s, py::return_value_policy::reference);
        },
        py::arg("name"));

    m.def(
        "op_registry_all",
        []() {
            std::vector<const OpSchema*> all = OpRegistry::all();
            return all;
        },
        py::return_value_policy::reference);

    m.def("op_registry_size", &OpRegistry::size);
}

}  // namespace lucid::bindings
