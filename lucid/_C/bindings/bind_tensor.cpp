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

#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/GradMode.h"
#include "../core/MemoryStats.h"
#include "../core/TensorImpl.h"
#include "../core/Storage.h"
#include "../backend/gpu/MetalAllocator.h"
#include "../backend/gpu/MlxBridge.h"
#include <mlx/ops.h>
#include <mlx/transforms.h>   // mlx::core::eval(std::vector<array>)

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
        .def_property_readonly("grad_fn",
                               [](const TensorImpl& t) -> std::shared_ptr<Node> {
                                   return t.grad_fn();
                               },
                               "The backward function that produced this tensor, or None "
                               "for leaf tensors.")
        .def_property_readonly("retains_grad",
                               [](const TensorImpl& t) { return t.retains_grad(); })
        .def("retain_grad_",
             [](TensorImpl& t) { t.set_retain_grad(true); },
             "Mark this tensor to retain its gradient during backward, even if "
             "it is not a leaf tensor.")
        .def_property_readonly("is_leaf", [](const TensorImpl& t) { return t.is_leaf(); })
        // version is a mutation counter; the autograd engine reads it to
        // detect in-place modifications to saved tensors.
        .def_property_readonly("version", [](const TensorImpl& t) { return t.version(); })
        .def_property_readonly(
            "is_metal_shared",
            [](const TensorImpl& t) -> bool {
                return storage_is_metal_shared(t.storage());
            },
            "True when the tensor's storage is a MTLResourceStorageModeShared "
            "allocation — simultaneously accessible from CPU and GPU without a "
            "memcpy.  Created by make_shared_tensor() or to_shared_storage().")
        .def("numel", &TensorImpl::numel)
        .def("nbytes", &TensorImpl::nbytes)
        .def("is_contiguous", &TensorImpl::is_contiguous)
        // data_as_python / grad_as_python return the underlying buffer as a
        // numpy array view without copying; the TensorImpl must outlive the
        // returned array (keep-alive is handled by pybind11's default policy
        // for member functions on shared_ptr holders).
        .def("data_as_python", &TensorImpl::data_as_python)
        .def("grad_as_python", &TensorImpl::grad_as_python)
        // NumPy-free interop — used by the Python serialization and repr
        // layers so ``import lucid`` works without numpy installed.
        .def("to_bytes", &TensorImpl::to_bytes,
             "Return the tensor data as a contiguous bytes blob "
             "(row-major).  GPU tensors are downloaded to CPU first.")
        .def_static("from_bytes", &TensorImpl::from_bytes,
                    py::arg("data"), py::arg("shape"), py::arg("dtype"),
                    py::arg("device") = Device::CPU,
                    py::arg("requires_grad") = false,
                    "Reconstruct a TensorImpl from a raw bytes blob + "
                    "metadata.  ``len(data)`` must equal "
                    "``prod(shape) * dtype_size(dtype)``.")
        .def("to_string", &TensorImpl::to_string,
             py::arg("precision") = 4, py::arg("threshold") = 1000,
             py::arg("edgeitems") = 3,
             "Render the tensor's data as a human-readable string.  "
             "Used by ``__repr__`` to avoid going through numpy.")
        .def("grad_to_tensor",
             [](const TensorImpl& t) -> std::shared_ptr<TensorImpl> {
                 return t.grad_to_tensor();
             },
             "Wrap the accumulated gradient as a fresh TensorImpl sharing "
             "the same Storage.  Returns None when no gradient has been "
             "accumulated.  Replaces the prior numpy round-trip in "
             "``Tensor.grad``.")
        .def("item", &TensorImpl::item,
             "Extract a single-element tensor's value as a Python scalar "
             "(int / float / bool / complex).  Throws when numel() != 1.")
        .def("grad_as_impl",
             [](const TensorImpl& t) -> std::shared_ptr<TensorImpl> {
                 return t.grad_as_impl();
             },
             "Return the gradient as a TensorImpl (set when backward was called with "
             "create_graph=True). Returns None if no graph-mode gradient is available.")
        .def("copy_from", &TensorImpl::copy_from)
        .def("zero_grad", &TensorImpl::zero_grad)
        .def("eval", &TensorImpl::eval,
             "Force evaluation of this tensor's lazy MLX graph.\n"
             "GPU tensors: calls mlx::core::eval() on the underlying array.\n"
             "CPU tensors: no-op.")
        .def("clone_with_grad",
             [](const TensorImpl& self, bool requires_grad) -> std::shared_ptr<TensorImpl> {
                 // Creates a new TensorImpl that SHARES the same Storage with a
                 // different requires_grad flag.  No data copy is made — only the
                 // autograd metadata differs.  This is the canonical way to flip
                 // requires_grad without going through a numpy round-trip.
                 return std::make_shared<TensorImpl>(
                     self.storage(), self.shape(), self.dtype(), self.device(), requires_grad);
             },
             py::arg("requires_grad"),
             "Return a new TensorImpl sharing the same storage but with a different "
             "requires_grad flag.  No data is copied.")
        .def("set_grad",
             [](TensorImpl& self, const std::shared_ptr<TensorImpl>& g) {
                 // Copies g's Storage into self's gradient slot, replacing any
                 // existing gradient.  Used by clip_grad and grad_scaler to write
                 // back scaled gradients without going through numpy.
                 Storage s = g->storage();          // copy the Storage variant
                 self.set_grad_storage(std::move(s));
             },
             py::arg("grad"),
             "Replace this tensor's gradient storage with a copy of grad's storage.\n"
             "Used internally by clip_grad and GradScaler; prefer .grad = tensor for "
             "normal use.");

    // to_shared_storage re-wraps the tensor's backing buffer as a Metal
    // MTLResourceStorageModeShared allocation.  The resulting tensor occupies
    // the same physical DRAM page as the GPU — use it with _run_metal_kernel
    // for zero-copy custom kernel dispatch.
    m.def(
        "to_shared_storage",
        [](const std::shared_ptr<TensorImpl>& t) -> std::shared_ptr<TensorImpl> {
            if (!t)
                throw std::invalid_argument("to_shared_storage: null tensor");
            // Metal allocation always goes through the GPU backend regardless
            // of the source tensor's device — the GPU backend's
            // to_shared_storage() handles both CpuStorage and GpuStorage
            // sources and is the only backend that can allocate
            // MTLResourceStorageModeShared buffers.
            auto& be = backend::Dispatcher::for_device(Device::GPU);
            Storage shared = be.to_shared_storage(t->storage(), t->shape());
            return std::make_shared<TensorImpl>(std::move(shared), t->shape(), t->dtype(),
                                                t->device(), t->requires_grad());
        },
        py::arg("tensor"),
        "Convert a tensor's storage to MTLResourceStorageModeShared.\n"
        "The returned tensor shares physical DRAM with the GPU — "
        "pass it to _run_metal_kernel for zero-copy custom Metal kernels.");

    // transfer_storage(tensor, device) — zero-copy device relabeling for
    // SharedStorage tensors.  Unlike to_shared_storage(), this never copies
    // data: it either wraps the Metal buffer as an MLX external array (GPU
    // target) or aliases cpu_ptr as a CpuStorage view (CPU target).
    // Raises if the source tensor is not already in SharedStorage.
    m.def(
        "transfer_storage",
        [](const std::shared_ptr<TensorImpl>& t,
           Device target_device) -> std::shared_ptr<TensorImpl> {
            if (!t)
                throw std::invalid_argument("transfer_storage: null tensor");
            const Storage& s = t->storage();
            if (!storage_is_metal_shared(s))
                throw std::invalid_argument(
                    "transfer_storage: source tensor must be in SharedStorage "
                    "(call to_shared_storage() or make_shared_tensor() first)");
            const SharedStorage& sh = storage_metal_shared(s);
            Storage new_storage;
            if (target_device == Device::GPU) {
                // Zero-copy: wrap as MLX external array pointing to the same
                // Metal buffer.  The owner shared_ptr is captured in the MLX
                // custom deleter so the Metal buffer stays alive.
                new_storage = Storage{gpu::shared_storage_to_gpu(sh, t->shape())};
            } else {
                // Zero-copy: alias cpu_ptr as a CpuStorage view.  The owner
                // shared_ptr is captured in the CpuStorage custom deleter.
                new_storage = Storage{sh.cpu_view()};
            }
            return std::make_shared<TensorImpl>(
                std::move(new_storage), t->shape(), t->dtype(),
                target_device, t->requires_grad());
        },
        py::arg("tensor"), py::arg("device"),
        "Zero-copy device relabeling for SharedStorage tensors.\n\n"
        "GPU target: wraps the Metal buffer as an MLX external array — no memcpy.\n"
        "CPU target: aliases cpu_ptr as a CpuStorage view — no memcpy.\n"
        "Raises ValueError if the source is not in SharedStorage.");

    // make_shared_tensor(shape, dtype, requires_grad) — allocate a
    // zero-filled tensor directly in MTLResourceStorageModeShared memory.
    // The result starts with device=CPU so normal CPU ops work immediately;
    // call transfer_storage(t, Device::GPU) for a zero-copy GPU view.
    m.def(
        "make_shared_tensor",
        [](const std::vector<std::int64_t>& shape,
           Dtype dtype,
           bool requires_grad) -> std::shared_ptr<TensorImpl> {
            std::size_t n = 1;
            for (auto d : shape)
                n *= static_cast<std::size_t>(d);
            const std::size_t nbytes = n * dtype_size(dtype);
            if (nbytes == 0) {
                // Empty tensor: fall back to ordinary CPU zeros.
                auto& be = backend::Dispatcher::for_device(Device::CPU);
                Storage s = be.zeros(Shape(shape), dtype);
                return std::make_shared<TensorImpl>(
                    std::move(s), Shape(shape), dtype, Device::CPU, requires_grad);
            }
            auto owned = gpu::make_metal_shared(nbytes);
            if (!owned.buf.cpu_ptr)
                throw std::runtime_error(
                    "make_shared_tensor: MTLBuffer allocation failed — "
                    "Metal may not be available on this device");
            // macOS guarantees MTLResourceStorageModeShared buffers are
            // zero-initialized; no explicit memset needed.
            SharedStorage ss;
            ss.cpu_ptr    = owned.buf.cpu_ptr;
            ss.mtl_handle = owned.buf.mtl_handle;
            ss.nbytes     = nbytes;
            ss.dtype      = dtype;
            ss.owner      = std::move(owned.owner);
            return std::make_shared<TensorImpl>(
                Storage{std::move(ss)}, Shape(shape), dtype,
                Device::CPU, requires_grad);
        },
        py::arg("shape"),
        py::arg("dtype")         = Dtype::F32,
        py::arg("requires_grad") = false,
        "Allocate a zero-filled tensor in MTLResourceStorageModeShared memory.\n\n"
        "The tensor is immediately readable on CPU (device=CPU) and can be\n"
        "transferred to GPU via transfer_storage() with no memcpy.\n"
        "Raises RuntimeError if Metal is not available.");

    // eval_tensors(list[TensorImpl]) — batch-evaluate multiple GPU tensors in
    // one mlx::core::eval() call.  CPU tensors in the list are silently ignored.
    // More efficient than calling .eval() individually because MLX schedules all
    // arrays in a single pass.
    m.def(
        "eval_tensors",
        [](const std::vector<std::shared_ptr<TensorImpl>>& tensors) {
            // Collect all GPU arrays, then evaluate in one batched MLX call.
            // mlx::core::eval(vector<array>) schedules them together —
            // significantly faster than calling arr->eval() individually.
            std::vector<mlx::core::array> arrays;
            arrays.reserve(tensors.size());
            for (const auto& t : tensors) {
                if (!t || t->device() != Device::GPU)
                    continue;
                const auto& gpu_st = std::get<GpuStorage>(t->storage());
                if (gpu_st.arr)
                    arrays.push_back(*gpu_st.arr);
            }
            if (!arrays.empty())
                mlx::core::eval(arrays);
        },
        py::arg("tensors"),
        "Batch-evaluate GPU tensors in one mlx::core::eval() call.\n"
        "CPU tensors are silently ignored.  No-op when all tensors are on CPU.");

    // eval_gpu(impl) — evaluate a single GPU TensorImpl in-place.
    // Faster than eval_tensors([impl]) for the single-tensor case because it
    // avoids Python list creation and the pybind11 vector conversion overhead
    // (~25 µs on hot paths).  Used by the benchmark's timing loop and by any
    // code that needs to force GPU computation for a single intermediate tensor.
    m.def(
        "eval_gpu",
        [](const std::shared_ptr<TensorImpl>& t) {
            if (!t || t->device() != Device::GPU)
                return;
            const auto& gs = std::get<GpuStorage>(t->storage());
            if (gs.arr)
                gs.arr->eval();
        },
        py::arg("tensor"),
        "Force evaluation of a single GPU tensor (no-op for CPU tensors).\n"
        "Faster than eval_tensors([t]) for the single-tensor hot path.");
}

}  // namespace lucid::bindings
