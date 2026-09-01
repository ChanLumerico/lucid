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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

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
}

}  // namespace lucid::bindings
