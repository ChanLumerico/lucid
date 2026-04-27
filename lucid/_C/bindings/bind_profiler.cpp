#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Profiler.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_profiler(py::module_& m) {
    py::class_<OpEvent>(m, "OpEvent")
        .def_readonly("name", &OpEvent::name)
        .def_readonly("device", &OpEvent::device)
        .def_readonly("dtype", &OpEvent::dtype)
        .def_readonly("shape", &OpEvent::shape)
        .def_readonly("time_ns", &OpEvent::time_ns)
        .def_readonly("memory_delta_bytes", &OpEvent::memory_delta_bytes)
        .def_readonly("flops", &OpEvent::flops)
        .def("__repr__", [](const OpEvent& e) {
            return "OpEvent(name='" + e.name + "', time_ns=" +
                   std::to_string(e.time_ns) + ", flops=" +
                   std::to_string(e.flops) + ")";
        });

    py::class_<Profiler>(m, "Profiler")
        .def(py::init<>())
        .def("start", &Profiler::start)
        .def("stop", &Profiler::stop)
        .def("clear", &Profiler::clear)
        .def_property_readonly("is_active", &Profiler::is_active)
        .def_property_readonly("events", &Profiler::events);

    m.def("set_current_profiler", &set_current_profiler,
          py::arg("profiler").none(true));
    m.def("current_profiler", &current_profiler,
          py::return_value_policy::reference);
}

}  // namespace lucid::bindings
