#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "../core/Error.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_errors(py::module_& m) {
    static py::object lucid_error_cls =
        py::exception<LucidError>(m, "LucidError", PyExc_RuntimeError);

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
