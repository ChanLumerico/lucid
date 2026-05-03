// lucid/_C/bindings/bind_tensor.cpp
//
// Exposes the two fundamental value-type APIs of the Lucid engine to Python:
//
//   register_core()   — Device enum, Dtype enum, dtype helpers, NoGradGuard
//                       context manager, global grad-mode toggle, and
//                       MemoryStats / MemoryTracker.
//
//   register_tensor_impl() — TensorImpl (Python alias: Tensor), the central
//                            tensor type.  TensorImpl is held by shared_ptr
//                            on both sides; pybind11 reference-counts via the
//                            shared_ptr holder.  Also registers to_shared_storage()
//                            which re-wraps storage as MTLResourceStorageModeShared
//                            for zero-copy Metal kernel dispatch.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../backend/Dispatcher.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/GradMode.h"
#include "../core/MemoryStats.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers Device, Dtype, dtype utilities, the no_grad context manager, and
// memory statistics types.  These must be present before TensorImpl and all
// op bindings that reference them.
void register_core(py::module_& m) {
    // Device and Dtype are exported as Python enums.  export_values() makes
    // the members available at module scope (e.g. engine.CPU) as well as on
    // the enum type (engine.Device.CPU).
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
    // dtype_name returns a string_view; the lambda converts to std::string so
    // pybind11 can copy it safely without a dangling pointer.
    m.def("dtype_name", [](Dtype d) { return std::string(dtype_name(d)); });

    // NoGradGuard implements the Python context-manager protocol.  The
    // __enter__ method returns a reference to self (return_value_policy::reference)
    // because Python's `with` binds the result but the guard object is already
    // owned by the surrounding scope.
    py::class_<NoGradGuard>(m, "NoGradGuard")
        .def(py::init<>())
        .def(
            "__enter__", [](NoGradGuard& g) -> NoGradGuard& { return g; },
            py::return_value_policy::reference)
        .def("__exit__",
             [](NoGradGuard&, py::object, py::object, py::object) { return py::none(); });

    m.def("grad_enabled", &GradMode::is_enabled);
    m.def("set_grad_enabled", &GradMode::set_enabled);

    // MemoryStats fields are read-only; the Python layer cannot mutate them.
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

// Registers TensorImpl as the Python-visible `TensorImpl` type and the
// standalone to_shared_storage() utility.
void register_tensor_impl(py::module_& m) {
    // TensorImpl is the sole tensor representation exposed to Python.  It is
    // held as shared_ptr<TensorImpl> on both the C++ and Python sides.  The
    // constructor delegates to TensorImpl::from_numpy, which copies the numpy
    // array data into a managed Storage buffer.
    py::class_<TensorImpl, std::shared_ptr<TensorImpl>>(m, "TensorImpl")
        .def(py::init([](py::array arr, Device device, bool requires_grad) {
                 return TensorImpl::from_numpy(std::move(arr), device, requires_grad);
             }),
             py::arg("data"), py::arg("device") = Device::CPU, py::arg("requires_grad") = false)
        // Shape and stride are returned as Python lists via py::cast.
        .def_property_readonly("shape", [](const TensorImpl& t) { return py::cast(t.shape()); })
        .def_property_readonly("stride", [](const TensorImpl& t) { return py::cast(t.stride()); })
        .def_property_readonly("dtype", [](const TensorImpl& t) { return t.dtype(); })
        .def_property_readonly("device", [](const TensorImpl& t) { return t.device(); })
        .def_property_readonly("requires_grad",
                               [](const TensorImpl& t) { return t.requires_grad(); })
        .def_property_readonly("is_leaf", [](const TensorImpl& t) { return t.is_leaf(); })
        // version is a mutation counter; the autograd engine reads it to
        // detect in-place modifications to saved tensors.
        .def_property_readonly("version", [](const TensorImpl& t) { return t.version(); })
        .def("numel", &TensorImpl::numel)
        .def("nbytes", &TensorImpl::nbytes)
        .def("is_contiguous", &TensorImpl::is_contiguous)
        // data_as_python / grad_as_python return the underlying buffer as a
        // numpy array view without copying; the TensorImpl must outlive the
        // returned array (keep-alive is handled by pybind11's default policy
        // for member functions on shared_ptr holders).
        .def("data_as_python", &TensorImpl::data_as_python)
        .def("grad_as_python", &TensorImpl::grad_as_python)
        .def("copy_from", &TensorImpl::copy_from)
        .def("zero_grad", &TensorImpl::zero_grad);

    // to_shared_storage re-wraps the tensor's backing buffer as a Metal
    // MTLResourceStorageModeShared allocation.  The resulting tensor occupies
    // the same physical DRAM page as the GPU — use it with _run_metal_kernel
    // for zero-copy custom kernel dispatch.
    m.def(
        "to_shared_storage",
        [](const std::shared_ptr<TensorImpl>& t) -> std::shared_ptr<TensorImpl> {
            if (!t)
                throw std::invalid_argument("to_shared_storage: null tensor");
            auto& be = backend::Dispatcher::for_device(t->device());
            Storage shared = be.to_shared_storage(t->storage(), t->shape());
            return std::make_shared<TensorImpl>(std::move(shared), t->shape(), t->dtype(),
                                                t->device(), t->requires_grad());
        },
        py::arg("tensor"),
        "Convert a tensor's storage to MTLResourceStorageModeShared.\n"
        "The returned tensor shares physical DRAM with the GPU — "
        "pass it to _run_metal_kernel for zero-copy custom Metal kernels.");
}

}  // namespace lucid::bindings
