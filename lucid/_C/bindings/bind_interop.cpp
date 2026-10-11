// lucid/_C/bindings/bind_interop.cpp
//
// Python surface for the Metal DLPack dialect (see interop/Dlpack.h).
//
// Capsule protocol, as the DLPack spec defines it and every consumer
// relies on: a fresh export is named ``"dltensor"``; whoever consumes it
// renames it to ``"used_dltensor"`` and takes over the block.  The
// destructor here therefore frees the block only if the name is still
// the original one — otherwise the consumer owns it and freeing would be
// a double free.
//
// The MLX queries at the end answer for the MLX the engine links (the pinned
// liblucid_mlx), not for any ``mlx`` Python package in the same process: the
// two are separate instances with separate buffer caches.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include <mlx/device.h>
#include <mlx/memory.h>
#include <mlx/version.h>

#include "../core/TensorImpl.h"
#include "../interop/Dlpack.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

constexpr const char* kFresh = "dltensor";
constexpr const char* kUsed = "used_dltensor";

void capsule_destructor(PyObject* capsule) {
    // Consumed capsules were renamed by their consumer, which now owns
    // the block.
    if (PyCapsule_IsValid(capsule, kUsed) != 0)
        return;
    if (PyCapsule_IsValid(capsule, kFresh) == 0)
        return;
    auto* managed =
        static_cast<lucid::interop::DLManagedTensor*>(PyCapsule_GetPointer(capsule, kFresh));
    if (managed != nullptr && managed->deleter != nullptr)
        managed->deleter(managed);
}

}  // namespace

void register_interop(py::module_& m) {
    m.def(
        "to_dlpack_metal",
        [](const TensorImplPtr& impl) -> py::object {
            lucid::interop::DLManagedTensor* managed = lucid::interop::metal_to_dlpack(impl);
            PyObject* capsule = PyCapsule_New(managed, kFresh, &capsule_destructor);
            if (capsule == nullptr) {
                if (managed->deleter != nullptr)
                    managed->deleter(managed);
                throw py::error_already_set();
            }
            return py::reinterpret_steal<py::object>(capsule);
        },
        py::arg("tensor"),
        "Export a Metal tensor as a kDLMetal DLPack capsule (zero-copy).\n\n"
        "The capsule's ``data`` field is the MTLBuffer handle, which is what "
        "MLX's own exporter emits and what its importer expects; NumPy cannot "
        "read a capsule of this device type, so the host path stays on the "
        "NumPy dialect. Raises for a CPU tensor or a dtype Metal has no "
        "storage for (float64).");

    m.def(
        "from_dlpack_metal",
        [](py::object capsule) -> TensorImplPtr {
            PyObject* raw = capsule.ptr();
            if (PyCapsule_IsValid(raw, kFresh) == 0) {
                throw std::invalid_argument(
                    "from_dlpack_metal: expected an unconsumed PyCapsule named "
                    "'dltensor'; a capsule can only be consumed once");
            }
            auto* managed =
                static_cast<lucid::interop::DLManagedTensor*>(PyCapsule_GetPointer(raw, kFresh));
            // Rename before adopting: from here on this side owns the
            // block, and the capsule's own destructor must keep its
            // hands off it even if the adoption below throws.
            PyCapsule_SetName(raw, kUsed);
            return lucid::interop::dlpack_to_metal(managed);
        },
        py::arg("capsule"),
        "Adopt a kDLMetal DLPack capsule as a Lucid GPU tensor (zero-copy).\n\n"
        "Consumes the capsule per the DLPack protocol — renames it to "
        "'used_dltensor' and takes ownership of the underlying block, whose "
        "deleter runs when the last reference to the resulting tensor's "
        "storage dies. Only row-major packed capsules are adopted; a strided "
        "one raises rather than quietly copying.");

    m.def(
        "mlx_runtime_version", [] { return std::string(mlx::core::version()); },
        "Version of the MLX library the engine has loaded (the build pinned in "
        "pyproject.toml [tool.lucid.mlx]), independent of any installed ``mlx`` "
        "Python package.");

    m.def(
        "metal_clear_cache", [] { mlx::core::clear_cache(); },
        "Release the engine's MLX buffer cache back to the OS. Leaves another "
        "MLX instance in the process (an ``mlx`` Python package) untouched.");

    m.def(
        "metal_cache_memory", [] { return mlx::core::get_cache_memory(); },
        "Bytes held in the engine's MLX buffer cache: reserved, not backing a "
        "live array.");

    // MLX's allocator counters, unlike MemoryTracker's (memory_stats), also
    // see buffers MLX allocates internally: graph intermediates, kernel
    // scratch, arrays held only by a lazy graph.
    m.def(
        "metal_active_memory", [] { return mlx::core::get_active_memory(); },
        "Bytes the engine's MLX allocator has handed out and not yet freed "
        "(excludes its cache).");

    m.def(
        "metal_peak_memory", [] { return mlx::core::get_peak_memory(); },
        "Highest metal_active_memory() since the process started or the last "
        "metal_reset_peak_memory().");

    m.def(
        "metal_reset_peak_memory", [] { mlx::core::reset_peak_memory(); },
        "Restart metal_peak_memory() from the current active memory.");

    m.def(
        "metal_device_info",
        [] { return mlx::core::device_info(mlx::core::Device(mlx::core::Device::gpu)); },
        "The Metal device description the engine's MLX reports (``device_name``, "
        "``architecture``, memory sizes, ...), as a dict of str or int values.");
}

}  // namespace lucid::bindings
