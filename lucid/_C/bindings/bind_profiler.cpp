// lucid/_C/bindings/bind_profiler.cpp
//
// Exposes the op-level profiling subsystem to Python.  Registers:
//   - OpEvent (read-only struct: name, device, dtype, shape, time_ns,
//     memory_delta_bytes, flops)
//   - Profiler (start/stop/clear/events/is_active)
//   - set_current_profiler() / current_profiler() — install or retrieve the
//     process-wide Profiler pointer consumed by the engine during op dispatch.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Profiler.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers the OpEvent record, the Profiler class, and the global profiler
// accessors.
void register_profiler(py::module_& m) {
    // OpEvent is a plain data record populated by the engine after each op.
    // All fields are read-only from Python; mutation goes through the C++ side.
    py::class_<OpEvent>(m, "OpEvent")
        .def_readonly("name", &OpEvent::name)
        .def_readonly("device", &OpEvent::device)
        .def_readonly("dtype", &OpEvent::dtype)
        .def_readonly("shape", &OpEvent::shape)
        .def_readonly("time_ns", &OpEvent::time_ns)
        .def_readonly("memory_delta_bytes", &OpEvent::memory_delta_bytes)
        .def_readonly("flops", &OpEvent::flops)
        .def("__repr__", [](const OpEvent& e) {
            return "OpEvent(name='" + e.name + "', time_ns=" + std::to_string(e.time_ns) +
                   ", flops=" + std::to_string(e.flops) + ")";
        });

    // Profiler accumulates OpEvents while active.  Python is the sole owner of
    // each Profiler instance; the engine holds only a non-owning pointer
    // installed via set_current_profiler().
    py::class_<Profiler>(m, "Profiler")
        .def(py::init<>())
        .def("start", &Profiler::start)
        .def("stop", &Profiler::stop)
        .def("clear", &Profiler::clear)
        .def_property_readonly("is_active", &Profiler::is_active)
        .def_property_readonly("events", &Profiler::events);

    // set_current_profiler accepts None to detach the active profiler.  The
    // engine holds a raw pointer, so Python must not destroy the Profiler
    // object while it is installed (keep-alive is the caller's responsibility).
    m.def("set_current_profiler", &set_current_profiler, py::arg("profiler").none(true));
    // current_profiler returns a borrowed reference; Python must not delete it.
    m.def("current_profiler", &current_profiler, py::return_value_policy::reference);
}

}  // namespace lucid::bindings
