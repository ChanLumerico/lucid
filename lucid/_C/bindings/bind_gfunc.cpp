// lucid/_C/bindings/bind_gfunc.cpp
//
// Registers tensor-creation ("generator function") ops: zeros, ones, full,
// empty, eye, arange, linspace, diag, and the _like variants.  These ops
// produce new tensors from scalar or shape arguments rather than from existing
// tensors, so they do not have backward nodes and do not appear in the binary-
// or unary-op registration files.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Shape.h"
#include "../core/TensorImpl.h"
#include "../core/Dtype.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/utils/Flip.h"
#include "../ops/utils/MaskedSelect.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

// Converts a Python list of int64 dimensions to the internal Shape typedef.
Shape vec_to_shape(const std::vector<std::int64_t>& v) {
    return Shape(v.begin(), v.end());
}

}  // namespace

// Registers all tensor-creation ops on the top-level engine module.
void register_gfunc(py::module_& m) {
    // zeros, ones, full, and empty accept a Python list for shape; the lambda
    // converts it to Shape before calling the C++ op.
    m.def(
        "zeros",
        [](std::vector<std::int64_t> shape, Dtype dt, Device device, bool requires_grad) {
            return zeros_op(vec_to_shape(shape), dt, device, requires_grad);
        },
        py::arg("shape"), py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
        py::arg("requires_grad") = false);

    m.def(
        "ones",
        [](std::vector<std::int64_t> shape, Dtype dt, Device device, bool requires_grad) {
            return ones_op(vec_to_shape(shape), dt, device, requires_grad);
        },
        py::arg("shape"), py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
        py::arg("requires_grad") = false);

    m.def(
        "full",
        [](std::vector<std::int64_t> shape, double value, Dtype dt, Device device,
           bool requires_grad) {
            return full_op(vec_to_shape(shape), value, dt, device, requires_grad);
        },
        py::arg("shape"), py::arg("fill_value"), py::arg("dtype") = Dtype::F32,
        py::arg("device") = Device::CPU, py::arg("requires_grad") = false);

    m.def(
        "empty",
        [](std::vector<std::int64_t> shape, Dtype dt, Device device, bool requires_grad) {
            return empty_op(vec_to_shape(shape), dt, device, requires_grad);
        },
        py::arg("shape"), py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
        py::arg("requires_grad") = false);

    // eye, arange, linspace, and diag delegate directly to their C++ ops
    // without shape conversion because they take scalar dimension arguments.
    // eye: M=-1 means square (N×N); k is the diagonal offset.
    m.def("eye", &eye_op, py::arg("N"), py::arg("M") = -1, py::arg("k") = 0,
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("arange", &arange_op, py::arg("start"), py::arg("stop"), py::arg("step") = 1.0,
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("linspace", &linspace_op, py::arg("start"), py::arg("stop"), py::arg("num") = 50,
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("diag", &diag_op, py::arg("v"), py::arg("k") = 0);

    m.def("logspace", &logspace_op,
          py::arg("start"), py::arg("stop"), py::arg("num") = 50,
          py::arg("base") = 10.0,
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("scatter_add", &scatter_add_op,
          py::arg("base"), py::arg("indices"), py::arg("src"), py::arg("dim"));

    m.def("unfold_dim", &unfold_dim_op,
          py::arg("a"), py::arg("dim"), py::arg("size"), py::arg("step"));

    // _like ops infer shape and device from an existing tensor; only the
    // requires_grad flag may be overridden.
    m.def("zeros_like", &zeros_like_op, py::arg("a"), py::arg("requires_grad") = false);
    m.def("ones_like", &ones_like_op, py::arg("a"), py::arg("requires_grad") = false);
    m.def("empty_like", &empty_like_op, py::arg("a"), py::arg("requires_grad") = false);
    m.def("full_like", &full_like_op, py::arg("a"), py::arg("fill_value"),
          py::arg("requires_grad") = false);

    m.def("flip", &flip_op, py::arg("a"), py::arg("dims"),
          "Reverse tensor along the given dims. CPU: loop copy. GPU: take with reversed indices.");

    m.def("masked_select", &masked_select_op, py::arg("a"), py::arg("mask"),
          "Boolean masked selection: returns 1-D tensor of elements where mask==True.");

    m.def("astype", &astype_op, py::arg("a"), py::arg("dtype"),
          "Cast all elements to dtype. CPU: static_cast loop. GPU: mlx::core::astype.");
}

}  // namespace lucid::bindings
