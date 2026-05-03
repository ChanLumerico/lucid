// lucid/_C/bindings/bind_op_registry.cpp
//
// Exposes the global OpRegistry and OpSchema to Python for introspection and
// tooling.  Python code can iterate all registered ops, check their AMP policy,
// verify input/output arity, and use `internal` to determine which ops should
// appear in generated stubs.  None of these bindings are required at op
// dispatch time; they are used by BindingGen and test infrastructure.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/OpRegistry.h"
#include "../core/OpSchema.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers the OpSchema read-only descriptor and the three registry query
// functions: op_lookup, op_registry_all, op_registry_size.
void register_op_registry(py::module_& m) {
    // OpSchema contains only compile-time-constant string_views and plain POD
    // fields.  The `name` and `determinism_note` fields are string_view, so
    // they are copied into std::string to avoid dangling references in Python.
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
        // `internal` is true for ops that are implementation details and should
        // not appear in generated .pyi stubs or Python-facing documentation.
        .def_readonly("internal", &OpSchema::internal)
        .def("__repr__", [](const OpSchema& s) {
            return "OpSchema(name='" + std::string(s.name) +
                   "', version=" + std::to_string(s.version) +
                   ", input_arity=" + std::to_string(s.input_arity) + ")";
        });

    m.def("schema_hash", &schema_hash, py::arg("schema"));

    // op_lookup returns None if the name is not registered rather than raising
    // so callers can use a simple `if schema := engine.op_lookup(name)` idiom.
    // The returned pointer is borrowed from the static registry; reference
    // policy prevents pybind11 from trying to delete it.
    m.def(
        "op_lookup",
        [](std::string_view name) -> py::object {
            const auto* s = OpRegistry::lookup(name);
            if (!s)
                return py::none();
            return py::cast(s, py::return_value_policy::reference);
        },
        py::arg("name"));

    // op_registry_all returns all registered schemas as a list of borrowed
    // pointers.  The vector is constructed on demand and its elements are
    // pointers into the static registry — do not store them past the extension
    // module's lifetime.
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
