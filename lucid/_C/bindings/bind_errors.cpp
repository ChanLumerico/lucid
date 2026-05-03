// lucid/_C/bindings/bind_errors.cpp
//
// Maps every C++ exception type defined in core/Error.h to a corresponding
// Python exception class and installs a pybind11 exception translator so that
// C++ throws propagate across the language boundary as typed Python exceptions.
// All custom exception classes are subclasses of `LucidError` (itself a
// subclass of Python's `RuntimeError`) so that Python catch-all handlers for
// `RuntimeError` still work.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "../core/Error.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers the LucidError hierarchy as Python exception classes and installs
// the global exception translator.  Must be called before any op bindings so
// that errors thrown during subsequent registration are caught correctly.
void register_errors(py::module_& m) {
    // LucidError is the base; created via py::exception so pybind11 owns
    // the type object.  All sub-classes are created dynamically via the
    // Python type() builtin to avoid needing a separate py::exception<>
    // instantiation for each leaf type.
    static py::object lucid_error_cls =
        py::exception<LucidError>(m, "LucidError", PyExc_RuntimeError);

    // Creates a Python subclass of LucidError and exports it as m.<name>.
    // Static locals ensure the class objects survive the module's lifetime
    // without holding a Python reference on the stack.
    auto make_subclass = [&](const char* name) {
        py::object cls =
            py::module_::import("builtins")
                .attr("type")(py::str(name), py::make_tuple(lucid_error_cls), py::dict());
        m.attr(name) = cls;
        return cls;
    };

    static py::object oom_cls = make_subclass("OutOfMemory");
    static py::object shape_mismatch_cls = make_subclass("ShapeMismatch");
    static py::object dtype_mismatch_cls = make_subclass("DtypeMismatch");
    static py::object device_mismatch_cls = make_subclass("DeviceMismatch");
    static py::object version_mismatch_cls = make_subclass("VersionMismatch");
    static py::object gpu_unavailable_cls = make_subclass("GpuNotAvailable");
    static py::object index_error_cls = make_subclass("IndexError");
    static py::object not_implemented_cls = make_subclass("NotImplementedError");

    // The translator is called for every active exception that crosses the
    // C++/Python boundary.  More-derived types are checked first so that they
    // shadow the base LucidError catch at the end.
    py::register_exception_translator([](std::exception_ptr p) {
        if (!p)
            return;
        try {
            std::rethrow_exception(p);
        } catch (const OutOfMemory& e) {
            PyErr_SetString(oom_cls.ptr(), e.what());
        } catch (const ShapeMismatch& e) {
            PyErr_SetString(shape_mismatch_cls.ptr(), e.what());
        } catch (const DtypeMismatch& e) {
            PyErr_SetString(dtype_mismatch_cls.ptr(), e.what());
        } catch (const DeviceMismatch& e) {
            PyErr_SetString(device_mismatch_cls.ptr(), e.what());
        } catch (const VersionMismatch& e) {
            PyErr_SetString(version_mismatch_cls.ptr(), e.what());
        } catch (const GpuNotAvailable& e) {
            PyErr_SetString(gpu_unavailable_cls.ptr(), e.what());
        } catch (const IndexError& e) {
            PyErr_SetString(index_error_cls.ptr(), e.what());
        } catch (const NotImplementedError& e) {
            PyErr_SetString(not_implemented_cls.ptr(), e.what());
        } catch (const LucidError& e) {
            PyErr_SetString(lucid_error_cls.ptr(), e.what());
        }
    });
}

}  // namespace lucid::bindings
