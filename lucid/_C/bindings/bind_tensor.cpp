#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/GradMode.h"
#include "../core/MemoryStats.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_core(py::module_& m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::enum_<Dtype>(m, "Dtype")
        .value("Bool", Dtype::Bool)
        .value("I8", Dtype::I8)
        .value("I16", Dtype::I16)
        .value("I32", Dtype::I32)
        .value("I64", Dtype::I64)
        .value("F16", Dtype::F16)
        .value("F32", Dtype::F32)
        .value("F64", Dtype::F64)
        .value("C64", Dtype::C64)
        .export_values();
    m.def("dtype_size", &dtype_size);
    m.def("dtype_name", [](Dtype d) { return std::string(dtype_name(d)); });

    py::class_<NoGradGuard>(m, "NoGradGuard")
        .def(py::init<>())
        .def(
            "__enter__", [](NoGradGuard& g) -> NoGradGuard& { return g; },
            py::return_value_policy::reference)
        .def("__exit__",
             [](NoGradGuard&, py::object, py::object, py::object) { return py::none(); });

    m.def("grad_enabled", &GradMode::is_enabled);
    m.def("set_grad_enabled", &GradMode::set_enabled);

    // P2 — memory accounting public API.
    py::class_<MemoryStats>(m, "MemoryStats")
        .def_readonly("current_bytes", &MemoryStats::current_bytes)
        .def_readonly("peak_bytes", &MemoryStats::peak_bytes)
        .def_readonly("alloc_count", &MemoryStats::alloc_count)
        .def_readonly("free_count", &MemoryStats::free_count)
        .def("__repr__", [](const MemoryStats& s) {
            return "MemoryStats(current_bytes=" + std::to_string(s.current_bytes) +
                   ", peak_bytes=" + std::to_string(s.peak_bytes) +
                   ", alloc_count=" + std::to_string(s.alloc_count) +
                   ", free_count=" + std::to_string(s.free_count) + ")";
        });
    m.def("memory_stats", &MemoryTracker::get_stats, py::arg("device"));
    m.def("reset_peak_memory_stats", &MemoryTracker::reset_peak, py::arg("device"));
}

void register_tensor_impl(py::module_& m) {
    py::class_<TensorImpl, std::shared_ptr<TensorImpl>>(m, "TensorImpl")
        .def(py::init([](py::array arr, Device device, bool requires_grad) {
                 return TensorImpl::from_numpy(std::move(arr), device, requires_grad);
             }),
             py::arg("data"), py::arg("device") = Device::CPU, py::arg("requires_grad") = false)
        .def_property_readonly("shape", [](const TensorImpl& t) { return py::cast(t.shape()); })
        .def_property_readonly("stride", [](const TensorImpl& t) { return py::cast(t.stride()); })
        .def_property_readonly("dtype", [](const TensorImpl& t) { return t.dtype(); })
        .def_property_readonly("device", [](const TensorImpl& t) { return t.device(); })
        .def_property_readonly("requires_grad",
                               [](const TensorImpl& t) { return t.requires_grad(); })
        .def_property_readonly("is_leaf", [](const TensorImpl& t) { return t.is_leaf(); })
        .def_property_readonly("version", [](const TensorImpl& t) { return t.version(); })
        .def("numel", &TensorImpl::numel)
        .def("nbytes", &TensorImpl::nbytes)
        .def("is_contiguous", &TensorImpl::is_contiguous)
        .def("data_as_python", &TensorImpl::data_as_python)
        .def("grad_as_python", &TensorImpl::grad_as_python)
        .def("copy_from", &TensorImpl::copy_from)
        .def("zero_grad", &TensorImpl::zero_grad);
}

}  // namespace lucid::bindings
